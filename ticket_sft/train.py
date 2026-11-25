"""LoRA supervised fine-tuning with TRL's SFTTrainer and PEFT.

Loss is computed only on the assistant completion. The published run is the
CPU float32 config in ticket_sft.config.TrainConfig. modal_train.py passes a
different config and was not executed.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import json
import logging
import math
import time
from datetime import datetime, timezone

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from ticket_sft.config import ROOT, TrainConfig
from ticket_sft.data import read_jsonl, sha256_file
from ticket_sft.logging_utils import setup_logging
from ticket_sft.prompts import (
    build_messages,
    render_prompt,
    render_training_text,
    system_prompt_sha256,
)
from ticket_sft.runtime_info import runtime_info
from ticket_sft.schema import canonical_json

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _loss_history(log_history: list[dict]) -> list[dict]:
    by_step = {}
    for entry in log_history:
        if "loss" not in entry or "step" not in entry:
            continue
        loss = float(entry["loss"])
        step = int(entry["step"])
        point = {"step": step, "loss": loss}
        if "epoch" in entry and entry["epoch"] is not None:
            point["epoch"] = float(entry["epoch"])
        by_step[step] = point
    return [by_step[step] for step in sorted(by_step)]


def _verify_completion_mask(tokenizer, prepared, examples: list[dict], max_length: int) -> dict:
    if len(prepared) != len(examples):
        raise RuntimeError(
            f"trainer dataset has {len(prepared)} rows, train file has {len(examples)}. "
            "Rows were dropped or reordered, so the loss mask was not checked."
        )
    supervised = 0
    total = 0
    lengths = []
    for row, example in zip(prepared, examples):
        labels = [int(token) for token in row["labels"]]
        input_ids = [int(token) for token in row["input_ids"]]
        if len(labels) != len(input_ids):
            raise RuntimeError(f"{example['id']} labels and input_ids differ in length")
        if len(input_ids) >= max_length:
            raise RuntimeError(
                f"{example['id']} is {len(input_ids)} tokens, at the max_length cap {max_length}. "
                "Refusing to train because the completion may have been truncated."
            )
        if labels[0] != -100:
            raise RuntimeError(f"{example['id']} does not mask the prompt")
        supervised_ids = [token for token in labels if token != -100]
        if not supervised_ids:
            raise RuntimeError(f"{example['id']} has no supervised tokens")
        decoded = tokenizer.decode(supervised_ids, skip_special_tokens=True)
        gold = canonical_json(example["label"])
        if gold not in decoded:
            raise RuntimeError(
                f"{example['id']} supervised text does not contain the gold JSON.\n"
                f"gold: {gold}\ndecoded: {decoded}"
            )
        supervised += len(supervised_ids)
        total += len(labels)
        lengths.append(len(input_ids))
    fraction = supervised / total
    if not 0.05 <= fraction <= 0.6:
        raise RuntimeError(f"supervised token fraction {fraction:.3f} is outside 0.05..0.6")
    return {
        "supervised_token_fraction": fraction,
        "max_sequence_length_seen": max(lengths),
        "mean_sequence_length": sum(lengths) / len(lengths),
        "truncated_examples": 0,
    }


def _write_format_example(tokenizer, example: dict) -> None:
    path = ROOT / "results" / "format_example.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    prompt = render_prompt(tokenizer, example["message"])
    full = render_training_text(tokenizer, example["message"], example["label"])
    gold = canonical_json(example["label"])
    path.write_text(
        "\n".join(
            [
                f"Example id: {example['id']}",
                "The loss is applied to the assistant completion only.",
                "",
                "PROMPT",
                prompt,
                "",
                "FULL TRAINING TEXT",
                full,
                "",
                "GOLD JSON",
                gold,
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_training(config: TrainConfig | None = None) -> dict:
    config = TrainConfig() if config is None else config
    if config.dtype not in DTYPES:
        raise ValueError(f"unsupported dtype {config.dtype}")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if config.device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device {config.device}")

    setup_logging(config.log_path)
    started = datetime.now(timezone.utc)
    script_clock = time.perf_counter()
    set_seed(config.seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    logging.info("train config model=%s device=%s dtype=%s seed=%s", config.model_id, config.device, config.dtype, config.seed)

    examples = read_jsonl(config.train_file)
    if not examples:
        raise RuntimeError(f"no training rows in {config.train_file}")

    logging.info("loading %s", config.model_id)
    load_clock = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_id, dtype=DTYPES[config.dtype])
    model.to(torch.device(config.device))
    model.config.use_cache = False
    load_seconds = time.perf_counter() - load_clock
    logging.info("model load took %.1fs", load_seconds)

    _write_format_example(tokenizer, examples[0])
    rows = []
    for example in examples:
        messages = build_messages(example["message"], example["label"])
        rows.append({"prompt": messages[:-1], "completion": messages[-1:]})
    dataset = Dataset.from_list(rows)

    microbatches = config.per_device_train_batch_size * config.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(examples) / microbatches)
    if float(config.epochs).is_integer():
        planned_steps = steps_per_epoch * int(config.epochs)
    else:
        planned_steps = max(1, math.ceil(steps_per_epoch * config.epochs))
    if config.max_steps_limit is not None:
        planned_steps = config.max_steps_limit
    warmup_steps = max(0, int(round(config.warmup_ratio * planned_steps)))

    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.target_modules),
    )
    args = SFTConfig(
        output_dir=str(config.trainer_dir),
        seed=config.seed,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps_limit if config.max_steps_limit is not None else -1,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        logging_first_step=True,
        logging_strategy="steps",
        save_strategy="no",
        eval_strategy="no",
        fp16=config.dtype == "float16" and config.device == "cuda",
        bf16=config.dtype == "bfloat16" and config.device == "cuda",
        use_cpu=config.device == "cpu",
        max_length=config.max_length,
        packing=False,
        padding_free=False,
        completion_only_loss=True,
        assistant_only_loss=False,
        shuffle_dataset=False,
        report_to="none",
        dataset_num_proc=1,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        gradient_checkpointing=config.gradient_checkpointing,
        disable_tqdm=True,
        remove_unused_columns=False,
    )
    logging.info("building SFTTrainer and checking the completion mask")
    prepare_clock = time.perf_counter()
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )
    mask_stats = _verify_completion_mask(tokenizer, trainer.train_dataset, examples, config.max_length)
    prepare_seconds = time.perf_counter() - prepare_clock
    logging.info(
        "supervised token fraction %.3f max length %s mean length %.1f",
        mask_stats["supervised_token_fraction"],
        mask_stats["max_sequence_length_seen"],
        mask_stats["mean_sequence_length"],
    )
    trainable = sum(parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in trainer.model.parameters())
    logging.info("trainable parameters %s / %s", trainable, total)

    logging.info("starting trainer.train()")
    train_clock = time.perf_counter()
    train_result = trainer.train()
    train_wall_seconds = time.perf_counter() - train_clock
    history = _loss_history(trainer.state.log_history)
    if not history:
        raise RuntimeError("trainer produced no logged losses")
    if any(not math.isfinite(point["loss"]) for point in history):
        raise RuntimeError("non-finite training loss")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    trainer_runtime = float(train_result.metrics.get("train_runtime", train_wall_seconds))
    summary = {
        "status": "measured",
        "model_id": config.model_id,
        "seed": config.seed,
        "max_steps_limit": config.max_steps_limit,
        "epochs_configured": config.epochs,
        "optimizer_steps": int(trainer.state.global_step),
        "train_examples": len(examples),
        "train_wall_seconds": train_wall_seconds,
        "trainer_train_runtime_seconds": trainer_runtime,
        "script_wall_seconds": time.perf_counter() - script_clock,
        "load_seconds": load_seconds,
        "prepare_seconds": prepare_seconds,
        "final_logged_loss": history[-1]["loss"],
        "min_logged_loss": min(point["loss"] for point in history),
        "loss_history": history,
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "supervised_token_fraction": mask_stats["supervised_token_fraction"],
        "max_sequence_length_seen": mask_stats["max_sequence_length_seen"],
        "mean_sequence_length": mask_stats["mean_sequence_length"],
        "truncated_examples": 0,
        "lora": {
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": list(config.target_modules),
        },
        "hyperparameters": {
            "learning_rate": config.learning_rate,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_length": config.max_length,
            "weight_decay": config.weight_decay,
            "warmup_ratio": config.warmup_ratio,
            "warmup_steps": warmup_steps,
            "logging_steps": config.logging_steps,
            "dtype": config.dtype,
            "device": config.device,
            "gradient_checkpointing": config.gradient_checkpointing,
        },
        "cuda_available": bool(torch.cuda.is_available()),
        "system_prompt_sha256": system_prompt_sha256(),
        "data_sha256": {config.train_file.name: sha256_file(config.train_file)},
        "runtime": runtime_info(),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    config.summary_path.parent.mkdir(parents=True, exist_ok=True)
    config.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logging.info(
        "finished steps=%s final_loss=%.4f train_wall=%.1fs",
        summary["optimizer_steps"],
        summary["final_logged_loss"],
        train_wall_seconds,
    )
    return summary


def main() -> None:
    run_training(TrainConfig())


if __name__ == "__main__":
    main()
