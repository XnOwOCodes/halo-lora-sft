"""Fail if the README, metrics JSON, predictions, and loss curve disagree.

This does not judge whether LoRA helped. It checks that the published numbers
are the ones produced by the scorer from the saved predictions, and that those
predictions are scored against the committed eval sets.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

from ticket_sft.config import (
    ROOT,
    BASELINE_METRICS_PATH,
    BASELINE_PREDICTIONS_PATH,
    EPOCHS,
    EVAL_IID_PATH,
    EVAL_OOD_PATH,
    GRAD_ACCUM,
    KEYWORD_METRICS_PATH,
    KEYWORD_PREDICTIONS_PATH,
    LEARNING_RATE,
    LOG_DIR,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LOSS_CURVE_PATH,
    MANIFEST_PATH,
    MAX_LENGTH,
    MAX_NEW_TOKENS,
    MODEL_ID,
    POST_METRICS_PATH,
    POST_PREDICTIONS_PATH,
    README_PATH,
    RESULTS_END,
    RESULTS_START,
    RESULTS_TABLE_PATH,
    RUN_META_PATH,
    SEED,
    TARGET_MODULES,
    TRAIN_PATH,
    TRAIN_SUMMARY_PATH,
    BATCH_SIZE,
)
from ticket_sft.data import read_jsonl, sha256_file
from ticket_sft.metrics import aggregate_scores, read_predictions, score_prediction
from ticket_sft.prompts import system_prompt_sha256
from ticket_sft.report import render_results_markdown
from ticket_sft.schema import KEYS, canonical_json

COUNT_FIELDS = (
    "n",
    "strict_json_correct",
    "schema_valid_correct",
    "canonical_exact_correct",
    "exact_match_correct",
    "field_correct_slots",
    "field_slots",
)


def fail(message: str) -> None:
    raise SystemExit(message)


def _load(path: Path) -> dict:
    if not path.exists():
        fail(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _readme_block(text: str) -> str:
    if RESULTS_START not in text or RESULTS_END not in text:
        fail("README is missing results markers")
    _, rest = text.split(RESULTS_START, 1)
    body, _ = rest.split(RESULTS_END, 1)
    return body.strip()


def _check_png(path: Path) -> None:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        fail(f"{path} is not a PNG")
    if len(data) < 2000:
        fail(f"{path} is unexpectedly small ({len(data)} bytes)")


def _check_predictions(metrics: dict, predictions_path: Path, expected_which: str) -> None:
    if metrics.get("status") != "measured":
        fail(f"{predictions_path} metrics status is not measured")
    if metrics.get("which") != expected_which:
        fail(f"{predictions_path} which is {metrics.get('which')}")
    file_bytes = predictions_path.read_bytes()
    digest = hashlib.sha256(file_bytes).hexdigest()
    if digest != metrics.get("predictions_sha256"):
        fail(f"{predictions_path} hash does not match the metrics file")
    rows = read_predictions(predictions_path)
    by_split = {"iid": [], "ood": []}
    for row in rows:
        by_split[row["split"]].append(row)
    data_files = {"iid": EVAL_IID_PATH, "ood": EVAL_OOD_PATH}
    for split, examples in ((name, read_jsonl(path)) for name, path in data_files.items()):
        preds = by_split[split]
        if len(preds) != len(examples):
            fail(f"{split} prediction count {len(preds)} != data count {len(examples)}")
        recomputed = []
        for pred, example in zip(preds, examples):
            if pred["id"] != example["id"]:
                fail(f"prediction id {pred['id']} != data id {example['id']}")
            if pred["gold"] != example["label"]:
                fail(f"{pred['id']} gold does not match the data file")
            judgment = score_prediction(example["label"], pred["raw"])
            for flag in ("strict_json", "schema_valid", "canonical_exact", "exact_match"):
                if pred[flag] != judgment[flag]:
                    fail(f"{pred['id']} stored {flag} does not match the scorer")
            if pred["fields"] != judgment["fields"]:
                fail(f"{pred['id']} stored fields do not match the scorer")
            recomputed.append(judgment)
        fresh = aggregate_scores(recomputed)
        stored = metrics["splits"][split]
        for field in COUNT_FIELDS:
            if stored[field] != fresh[field]:
                fail(f"{expected_which} {split} {field} is {stored[field]}, scorer says {fresh[field]}")
        for key in KEYS:
            if stored["per_field_correct"][key] != fresh["per_field_correct"][key]:
                fail(f"{expected_which} {split} {key} count drifted")
        if expected_which != "keyword_baseline" and stored.get("file_sha256") != sha256_file(
            data_files[split]
        ):
            fail(f"{split} file hash drifted")


def _check_train(summary: dict) -> None:
    if summary.get("status") != "measured":
        fail("train summary status is not measured")
    if summary["model_id"] != MODEL_ID:
        fail(f"train model is {summary['model_id']}")
    if summary["seed"] != SEED:
        fail("train seed changed")
    if summary["max_steps_limit"] is not None:
        fail("published run must not set max_steps_limit")
    if summary["cuda_available"] is not False:
        fail("published train summary does not say cuda_available false")
    if summary["hyperparameters"]["device"] != "cpu":
        fail("published train device is not cpu")
    if summary["hyperparameters"]["dtype"] != "float32":
        fail("published train dtype is not float32")
    if float(summary["epochs_configured"]) != float(EPOCHS):
        fail("epoch count changed")
    train_rows = read_jsonl(TRAIN_PATH)
    if summary["train_examples"] != len(train_rows):
        fail("train_examples does not match train.jsonl")
    per_epoch = math.ceil(
        len(train_rows)
        / (
            summary["hyperparameters"]["per_device_train_batch_size"]
            * summary["hyperparameters"]["gradient_accumulation_steps"]
        )
    )
    if not float(summary["epochs_configured"]).is_integer():
        fail("epoch check expects a whole number of epochs")
    expected_steps = per_epoch * int(summary["epochs_configured"])
    if summary["optimizer_steps"] != expected_steps:
        fail(f"optimizer steps {summary['optimizer_steps']} != expected {expected_steps}")
    if summary["truncated_examples"] != 0:
        fail("training truncated examples")
    history = summary["loss_history"]
    if len(history) != summary["optimizer_steps"]:
        fail("loss history does not have one point per optimizer step")
    losses = []
    previous = 0
    for point in history:
        loss = point["loss"]
        if not isinstance(loss, (int, float)) or not math.isfinite(loss):
            fail(f"non-finite loss {loss}")
        if point["step"] <= previous:
            fail("loss steps are not increasing")
        previous = point["step"]
        losses.append(float(loss))
    if history[-1]["step"] != summary["optimizer_steps"]:
        fail("loss history does not include the final optimizer step")
    if summary["final_logged_loss"] != history[-1]["loss"]:
        fail("final_logged_loss is not the last logged loss")
    if summary["min_logged_loss"] != min(point["loss"] for point in history):
        fail("min_logged_loss is not the minimum logged loss")
    fraction = summary["supervised_token_fraction"]
    if not 0.05 <= fraction <= 0.6:
        fail(f"supervised token fraction {fraction} is outside the expected mask range")
    trainable = summary["trainable_parameters"]
    total = summary["total_parameters"]
    if not 0 < trainable < total:
        fail("trainable parameter count is not a LoRA-sized subset")
    if trainable / total >= 0.1:
        fail("trainable fraction is too large for this LoRA setup")
    if summary["lora"]["r"] != LORA_R or summary["lora"]["alpha"] != LORA_ALPHA:
        fail("LoRA rank or alpha changed")
    if not math.isclose(float(summary["lora"]["dropout"]), float(LORA_DROPOUT), rel_tol=0, abs_tol=1e-12):
        fail("LoRA dropout changed")
    if tuple(summary["lora"]["target_modules"]) != tuple(TARGET_MODULES):
        fail("LoRA target modules changed")
    hyper = summary["hyperparameters"]
    if not math.isclose(float(hyper["learning_rate"]), float(LEARNING_RATE), rel_tol=0, abs_tol=1e-12):
        fail("learning rate changed")
    if hyper["per_device_train_batch_size"] != BATCH_SIZE:
        fail("batch size changed")
    if hyper["gradient_accumulation_steps"] != GRAD_ACCUM:
        fail("grad accumulation changed")
    if hyper["max_length"] != MAX_LENGTH:
        fail("max length changed")
    if summary["train_wall_seconds"] <= 0:
        fail("train wall time is not positive")
    if summary["system_prompt_sha256"] != system_prompt_sha256():
        fail("train prompt hash does not match prompts.py")
    if summary["data_sha256"]["train.jsonl"] != sha256_file(TRAIN_PATH):
        fail("train summary data hash does not match train.jsonl")
    example = ROOT / "results" / "format_example.txt"
    if not example.exists():
        fail("missing results/format_example.txt")
    text = example.read_text(encoding="utf-8")
    first = train_rows[0]
    if first["message"] not in text:
        fail("format example does not contain the first training message")
    if canonical_json(first["label"]) not in text:
        fail("format example does not contain the first gold JSON")
    if first["frame_id"] in text:
        fail("format example leaked a frame id")


def _check_eval_pair(baseline: dict, post: dict) -> None:
    for payload, which in ((baseline, "base"), (post, "adapter")):
        if payload["model_id"] != MODEL_ID:
            fail(f"{which} model id changed")
        if payload["cuda_available"] is not False:
            fail(f"{which} eval does not say cuda_available false")
        if payload["device"] != "cpu" or payload["dtype"] != "float32":
            fail(f"{which} eval is not the CPU float32 run")
        if payload["seed"] != SEED:
            fail(f"{which} seed changed")
        if payload["system_prompt_sha256"] != system_prompt_sha256():
            fail(f"{which} prompt hash changed")
        generation = payload["generation"]
        if generation["do_sample"] is not False or generation["num_beams"] != 1:
            fail(f"{which} generation is not greedy")
        if generation["max_new_tokens"] != MAX_NEW_TOKENS:
            fail(f"{which} max_new_tokens changed")
        if payload["elapsed_seconds"] <= 0:
            fail(f"{which} elapsed time is not positive")
    if baseline["generation"] != post["generation"]:
        fail("baseline and post generation configs differ")
    manifest = _load(MANIFEST_PATH)
    for split, filename in (("iid", "eval_iid.jsonl"), ("ood", "eval_ood.jsonl")):
        if baseline["splits"][split]["file_sha256"] != manifest["sha256"][filename]:
            fail(f"baseline {split} hash does not match the manifest")
        if post["splits"][split]["file_sha256"] != manifest["sha256"][filename]:
            fail(f"post {split} hash does not match the manifest")
    if baseline["adapter_path"] is not None:
        fail("baseline metrics name an adapter")
    if not post["adapter_path"]:
        fail("post metrics do not name an adapter")
    base_ids = [row["id"] for row in read_predictions(BASELINE_PREDICTIONS_PATH)]
    post_ids = [row["id"] for row in read_predictions(POST_PREDICTIONS_PATH)]
    if base_ids != post_ids:
        fail("baseline and post predictions are not the same examples in the same order")


def _check_manifest() -> None:
    manifest = _load(MANIFEST_PATH)
    for name, filename in (
        ("train", "train.jsonl"),
        ("iid", "eval_iid.jsonl"),
        ("ood", "eval_ood.jsonl"),
    ):
        path = MANIFEST_PATH.parent / filename
        if manifest["sha256"][filename] != sha256_file(path):
            fail(f"manifest hash for {filename} does not match the file")
        if manifest["counts"][name] != len(read_jsonl(path)):
            fail(f"manifest count for {name} does not match the file")
    if manifest["seed"] != SEED:
        fail("manifest seed changed")


def _check_readme() -> None:
    block = render_results_markdown().strip()
    readme = _readme_block(README_PATH.read_text(encoding="utf-8"))
    table = RESULTS_TABLE_PATH.read_text(encoding="utf-8").strip()
    if readme != block:
        fail("README results block does not match the metrics files")
    if table != block:
        fail("results/results_table.md does not match the metrics files")


def _check_run_meta(baseline: dict, post: dict, keyword: dict) -> None:
    meta = _load(RUN_META_PATH)
    if meta.get("status") != "measured":
        fail("run_meta status is not measured")
    expected = {}
    for split in ("iid", "ood"):
        expected[f"{split}_base"] = [
            baseline["splits"][split]["exact_match_correct"],
            baseline["splits"][split]["n"],
        ]
        expected[f"{split}_adapter"] = [
            post["splits"][split]["exact_match_correct"],
            post["splits"][split]["n"],
        ]
        expected[f"{split}_keyword"] = [
            keyword["splits"][split]["exact_match_correct"],
            keyword["splits"][split]["n"],
        ]
    if meta["exact_match"] != expected:
        fail("run_meta exact-match counts do not match the metrics files")
    for name, seconds in meta["stages_seconds"].items():
        if seconds <= 0:
            fail(f"stage {name} has non-positive time")


def _check_git_has_no_weights() -> None:
    tracked = subprocess.check_output(["git", "ls-files"], text=True)
    for line in tracked.splitlines():
        if line.startswith("artifacts/") or line.endswith(
            (".safetensors", ".bin", ".pt", ".pth", ".gguf")
        ):
            fail(f"weight file is tracked: {line}")


def main() -> None:
    for path in (
        LOG_DIR / "baseline_eval.log",
        LOG_DIR / "train.log",
        LOG_DIR / "post_eval.log",
        LOG_DIR / "keyword_baseline.log",
    ):
        if not path.exists() or path.stat().st_size == 0:
            fail(f"missing or empty log {path}")
    _check_png(LOSS_CURVE_PATH)
    baseline = _load(BASELINE_METRICS_PATH)
    post = _load(POST_METRICS_PATH)
    keyword = _load(KEYWORD_METRICS_PATH)
    summary = _load(TRAIN_SUMMARY_PATH)
    _check_predictions(baseline, BASELINE_PREDICTIONS_PATH, "base")
    _check_predictions(post, POST_PREDICTIONS_PATH, "adapter")
    _check_predictions(keyword, KEYWORD_PREDICTIONS_PATH, "keyword_baseline")
    _check_eval_pair(baseline, post)
    _check_train(summary)
    _check_manifest()
    _check_readme()
    _check_run_meta(baseline, post, keyword)
    _check_git_has_no_weights()
    print("results check passed")


if __name__ == "__main__":
    main()
