"""Greedy evaluation of the base model or a LoRA adapter.

The published CPU run calls this twice: once before training and once after.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from ticket_sft.config import (
    ROOT,
    ADAPTER_DIR,
    BASELINE_METRICS_PATH,
    BASELINE_PREDICTIONS_PATH,
    EVAL_IID_PATH,
    EVAL_OOD_PATH,
    LOG_DIR,
    MAX_NEW_TOKENS,
    MODEL_ID,
    POST_METRICS_PATH,
    POST_PREDICTIONS_PATH,
    SEED,
)
from ticket_sft.data import read_jsonl, sha256_file
from ticket_sft.logging_utils import setup_logging
from ticket_sft.metrics import (
    aggregate_scores,
    prediction_record,
    score_prediction,
    sha256_text,
    write_metrics,
    write_predictions,
)
from ticket_sft.prompts import render_prompt, system_prompt_sha256
from ticket_sft.runtime_info import runtime_info

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _support(examples: list[dict]) -> dict:
    return {
        "intent": dict(sorted(Counter(row["label"]["intent"] for row in examples).items())),
        "priority": dict(sorted(Counter(row["label"]["priority"] for row in examples).items())),
        "presence": dict(sorted(Counter(row["presence"] for row in examples).items())),
    }


def _generate_one(model, tokenizer, prompt: str, max_new_tokens: int, device: torch.device) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_evaluation(
    *,
    model_id: str,
    split_files: dict[str, Path],
    adapter_path: Path | None,
    dtype: str,
    device: str,
    max_new_tokens: int,
    seed: int,
) -> dict:
    if dtype not in DTYPES:
        raise ValueError(f"unsupported dtype {dtype}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device {device}")

    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    set_seed(seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

    logging.info("loading tokenizer and model %s dtype=%s device=%s", model_id, dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPES[dtype])
    adapter_summary = None
    if adapter_path is not None:
        from peft import PeftModel

        config_path = Path(adapter_path) / "adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"missing adapter config at {config_path}")
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        base_name = raw_config.get("base_model_name_or_path")
        if base_name not in {None, model_id}:
            raise RuntimeError(
                f"adapter base model is {base_name}, evaluation model is {model_id}"
            )
        adapter_summary = {
            "r": raw_config.get("r"),
            "lora_alpha": raw_config.get("lora_alpha"),
            "lora_dropout": raw_config.get("lora_dropout"),
            "target_modules": raw_config.get("target_modules"),
            "task_type": raw_config.get("task_type"),
            "base_model_name_or_path": base_name,
            "peft_type": raw_config.get("peft_type"),
        }
        model = PeftModel.from_pretrained(model, adapter_path)
        logging.info("loaded adapter %s", adapter_path)
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    model.config.use_cache = True

    split_metrics = {}
    rows = []
    logged_prompt = False
    for split_name, path in split_files.items():
        examples = read_jsonl(path)
        scored = []
        for index, example in enumerate(examples, start=1):
            prompt = render_prompt(tokenizer, example["message"])
            if not logged_prompt:
                logging.info("prompt example:\n%s", prompt)
                logged_prompt = True
            raw = _generate_one(model, tokenizer, prompt, max_new_tokens, torch_device)
            judgment = score_prediction(example["label"], raw)
            scored.append(judgment)
            rows.append(prediction_record(example, raw, judgment))
            preview = raw.replace("\n", " ")[:180]
            logging.info(
                "%s exact=%s strict=%s schema=%s raw=%s",
                example["id"],
                judgment["exact_match"],
                judgment["strict_json"],
                judgment["schema_valid"],
                preview,
            )
            if index % 10 == 0 or index == len(examples):
                logging.info("%s progress %s/%s", split_name, index, len(examples))
        metrics = aggregate_scores(scored)
        metrics["gold_support"] = _support(examples)
        metrics["file"] = path.name
        metrics["file_sha256"] = sha256_file(path)
        split_metrics[split_name] = metrics
        logging.info(
            "%s exact match %s/%s field slots %s/%s",
            split_name,
            metrics["exact_match_correct"],
            metrics["n"],
            metrics["field_correct_slots"],
            metrics["field_slots"],
        )

    elapsed = time.perf_counter() - clock
    info = runtime_info()
    payload = {
        "status": "measured",
        "which": "adapter" if adapter_path is not None else "base",
        "model_id": model_id,
        "adapter_path": None if adapter_path is None else _display_path(adapter_path),
        "adapter": adapter_summary,
        "seed": seed,
        "dtype": dtype,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
        },
        "system_prompt_sha256": system_prompt_sha256(),
        "splits": split_metrics,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "runtime": info,
    }
    return {"metrics": payload, "predictions": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the base model or a LoRA adapter.")
    parser.add_argument("--which", choices=("base", "adapter"), required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--adapter", type=Path, default=ADAPTER_DIR)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--iid", type=Path, default=EVAL_IID_PATH)
    parser.add_argument("--ood", type=Path, default=EVAL_OOD_PATH)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()

    if args.which == "base":
        metrics_path = args.metrics or BASELINE_METRICS_PATH
        predictions_path = args.predictions or BASELINE_PREDICTIONS_PATH
        log_path = args.log or LOG_DIR / "baseline_eval.log"
        adapter = None
    else:
        metrics_path = args.metrics or POST_METRICS_PATH
        predictions_path = args.predictions or POST_PREDICTIONS_PATH
        log_path = args.log or LOG_DIR / "post_eval.log"
        adapter = args.adapter

    setup_logging(log_path)
    result = run_evaluation(
        model_id=args.model,
        split_files={"iid": args.iid, "ood": args.ood},
        adapter_path=adapter,
        dtype=args.dtype,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    text = write_predictions(predictions_path, result["predictions"])
    result["metrics"]["predictions_sha256"] = sha256_text(text)
    write_metrics(metrics_path, result["metrics"])
    logging.info("wrote %s and %s", metrics_path, predictions_path)


if __name__ == "__main__":
    main()
