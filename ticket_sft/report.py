"""Build the results table from the measured JSON files and splice it into the README."""

from __future__ import annotations

import json

from ticket_sft.config import (
    BASELINE_METRICS_PATH,
    BASELINE_PREDICTIONS_PATH,
    KEYWORD_METRICS_PATH,
    LOSS_CURVE_PATH,
    MANIFEST_PATH,
    POST_METRICS_PATH,
    POST_PREDICTIONS_PATH,
    README_PATH,
    RESULTS_END,
    RESULTS_START,
    RESULTS_TABLE_PATH,
    TRAIN_SUMMARY_PATH,
)
from ticket_sft.metrics import read_predictions
from ticket_sft.schema import KEYS, canonical_json


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _count(correct: int, total: int) -> str:
    return f"{correct}/{total} ({100.0 * correct / total:.1f}%)"


def _direction(before: int, after: int) -> str:
    if after > before:
        return "higher"
    if after < before:
        return "lower"
    return "unchanged"


def _system_name(which: str, model_id: str | None) -> str:
    short = model_id.split("/")[-1] if model_id else ""
    if which == "base":
        return f"base {short}"
    if which == "adapter":
        return f"{short} + LoRA"
    if which == "keyword_baseline":
        return "keyword heuristic (train cues)"
    raise ValueError(which)


def _row(split: str, which: str, model_id: str | None, metrics: dict) -> str:
    fields = metrics["per_field_correct"]
    cells = [
        split,
        _system_name(which, model_id),
        _count(metrics["strict_json_correct"], metrics["n"]),
        _count(metrics["canonical_exact_correct"], metrics["n"]),
        _count(metrics["schema_valid_correct"], metrics["n"]),
        _count(metrics["exact_match_correct"], metrics["n"]),
        _count(metrics["field_correct_slots"], metrics["field_slots"]),
        _count(fields["intent"], metrics["n"]),
        _count(fields["order_id"], metrics["n"]),
        _count(fields["amount_usd"], metrics["n"]),
        _count(fields["priority"], metrics["n"]),
    ]
    return "| " + " | ".join(cells) + " |"


def _fence(text: str) -> str:
    truncated = False
    shown = text if text != "" else "(empty string)"
    if len(shown) > 500:
        shown = shown[:500]
        truncated = True
    shown = shown.replace("```", "'''")
    suffix = "\n[truncated; full text is in the predictions jsonl]" if truncated else ""
    return "```\n" + shown + suffix + "\n```"


def _fields(fields: dict) -> str:
    return ", ".join(f"{key}={'yes' if fields[key] else 'no'}" for key in KEYS)


def _first(rows: list[dict], split: str) -> dict:
    for row in rows:
        if row["split"] == split:
            return row
    raise KeyError(split)


def _example(title: str, base_row: dict, post_row: dict) -> str:
    if base_row["id"] != post_row["id"] or base_row["gold"] != post_row["gold"]:
        raise SystemExit("baseline and post predictions are not aligned")
    gold = canonical_json(base_row["gold"])
    return "\n".join(
        [
            f"{title} (`{base_row['id']}`), first row in the predictions file, not selected by hand:",
            "",
            f"Gold: `{gold}`",
            "",
            "Base model output:",
            "",
            _fence(base_row["raw"]),
            "",
            f"Base fields correct: {_fields(base_row['fields'])}.",
            "",
            "LoRA output:",
            "",
            _fence(post_row["raw"]),
            "",
            f"LoRA fields correct: {_fields(post_row['fields'])}.",
        ]
    )


def render_results_markdown() -> str:
    baseline = _load(BASELINE_METRICS_PATH)
    post = _load(POST_METRICS_PATH)
    keyword = _load(KEYWORD_METRICS_PATH)
    summary = _load(TRAIN_SUMMARY_PATH)
    manifest = _load(MANIFEST_PATH)
    base_rows = read_predictions(BASELINE_PREDICTIONS_PATH)
    post_rows = read_predictions(POST_PREDICTIONS_PATH)

    runtime = summary["runtime"]
    lora = summary["lora"]
    hyper = summary["hyperparameters"]
    iid_n = baseline["splits"]["iid"]["n"]
    ood_n = baseline["splits"]["ood"]["n"]
    iid_before = baseline["splits"]["iid"]["exact_match_correct"]
    iid_after = post["splits"]["iid"]["exact_match_correct"]
    ood_before = baseline["splits"]["ood"]["exact_match_correct"]
    ood_after = post["splits"]["ood"]["exact_match_correct"]
    trainable = summary["trainable_parameters"]
    total = summary["total_parameters"]
    targets = ", ".join(lora["target_modules"])
    if iid_n == ood_n:
        point = (
            f"One item on either {iid_n}-row eval split is {100.0 / iid_n:.1f} percentage points."
        )
    else:
        point = (
            f"One iid item is {100.0 / iid_n:.1f} points; one ood item is "
            f"{100.0 / ood_n:.1f} points."
        )
    cuda = "true" if summary["cuda_available"] else "false"
    header = [
        "These figures are computed from `results/*.json` and the prediction jsonl files.",
        "They are not typed in by hand. Exact match is semantic: recovered JSON must be",
        "schema-valid and every field must match the gold label.",
        "",
        (
            f"Hardware: {runtime['cpu_model']}, {runtime['cpu_count']} CPUs, "
            f"Python {runtime['python']}, torch {runtime['torch']}. "
            f"CUDA available: {cuda}. Seed {summary['seed']}."
        ),
        "",
        (
            f"Model: `{summary['model_id']}`. Train rows: {summary['train_examples']}. "
            f"Eval rows: iid {iid_n}, ood {ood_n}. "
            f"LoRA r={lora['r']}, alpha={lora['alpha']}, dropout={lora['dropout']}, "
            f"targets `{targets}`. "
            f"Trainable parameters: {trainable:,} / {total:,} "
            f"({100.0 * trainable / total:.2f}%). "
            f"Supervised tokens: {100.0 * summary['supervised_token_fraction']:.1f}% of training tokens. "
            f"Max sequence length seen: {summary['max_sequence_length_seen']} "
            f"(limit {hyper['max_length']}, truncated {summary['truncated_examples']})."
        ),
        "",
        (
            f"Training loop wall time: {summary['train_wall_seconds']:.1f}s. "
            f"Trainer-reported train_runtime: {summary['trainer_train_runtime_seconds']:.1f}s. "
            f"Optimizer steps: {summary['optimizer_steps']}. "
            f"Final logged training loss: {summary['final_logged_loss']:.4f}. "
            f"Minimum logged training loss: {summary['min_logged_loss']:.4f}. "
            f"Baseline eval wall time: {baseline['elapsed_seconds']:.1f}s. "
            f"Post-training eval wall time: {post['elapsed_seconds']:.1f}s."
        ),
        "",
        (
            "Generation is greedy "
            f"(`do_sample={str(baseline['generation']['do_sample']).lower()}`, "
            f"`max_new_tokens={baseline['generation']['max_new_tokens']}`). "
            f"{point} "
            "The keyword heuristic is not a neural model. It always emits canonical JSON, "
            "so its strict-JSON rate is 1 by construction."
        ),
        "",
        (
            f"In-distribution exact match is {_direction(iid_before, iid_after)} after LoRA "
            f"({iid_before}/{iid_n} to {iid_after}/{iid_n}). "
            f"Paraphrase exact match is {_direction(ood_before, ood_after)} after LoRA "
            f"({ood_before}/{ood_n} to {ood_after}/{ood_n})."
        ),
        "",
        "| Split | System | Strict JSON | Canonical JSON | Schema valid | Exact match | Field accuracy | Intent | Order id | Amount | Priority |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    table = []
    sources = (
        ("base", baseline),
        ("adapter", post),
        ("keyword_baseline", keyword),
    )
    for split in ("iid", "ood"):
        for which, payload in sources:
            table.append(_row(split, which, payload["model_id"], payload["splits"][split]))
    notes = [
        "",
        "Training loss is token cross-entropy on the supervised completion, not the exact-match score.",
        "Each plotted point is one logged optimizer step. The curve is not smoothed.",
        f"Loss curve file: `{LOSS_CURVE_PATH.relative_to(LOSS_CURVE_PATH.parents[1])}`.",
        "",
        (
            "Data sha256: "
            f"train `{manifest['sha256']['train.jsonl']}`, "
            f"iid `{manifest['sha256']['eval_iid.jsonl']}`, "
            f"ood `{manifest['sha256']['eval_ood.jsonl']}`."
        ),
        f"System prompt sha256 `{baseline['system_prompt_sha256']}`.",
        "",
        _example("In-distribution example", _first(base_rows, "iid"), _first(post_rows, "iid")),
        "",
        _example("Paraphrase example", _first(base_rows, "ood"), _first(post_rows, "ood")),
    ]
    return "\n".join(header + table + notes)


def splice_readme(readme: str, block: str) -> str:
    if RESULTS_START not in readme or RESULTS_END not in readme:
        raise SystemExit("README is missing results markers")
    pre, rest = readme.split(RESULTS_START, 1)
    _, post = rest.split(RESULTS_END, 1)
    return f"{pre}{RESULTS_START}\n{block.strip()}\n{RESULTS_END}{post}"


def write_report() -> str:
    block = render_results_markdown()
    RESULTS_TABLE_PATH.write_text(block.strip() + "\n", encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")
    README_PATH.write_text(splice_readme(readme, block), encoding="utf-8")
    return block


def main() -> None:
    write_report()
    print(RESULTS_TABLE_PATH)


if __name__ == "__main__":
    main()
