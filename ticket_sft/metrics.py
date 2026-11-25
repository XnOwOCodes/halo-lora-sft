"""Automatic metrics for ticket JSON.

exact_match is semantic: the recovered object is schema-valid and every field
matches the gold label. It does not require the bytes to equal canonical JSON.
canonical_exact is that stricter byte match.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ticket_sft.schema import KEYS, amounts_equal, canonical_json, schema_valid

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", flags=re.DOTALL)


def _load_dict(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(value, dict):
        return value
    return None


def _extract_candidate(text: str) -> str | None:
    fence = _FENCE_RE.search(text)
    if fence:
        return fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return None


def parse_model_output(text: str) -> tuple[bool, dict | None]:
    """Return (strict_json, recovered_object).

    strict_json is true only when the stripped generation itself is a JSON object.
    recovered_object also accepts one fenced block or the outer {...} span.
    """
    stripped = text.strip()
    strict_obj = _load_dict(stripped)
    recovered = strict_obj
    if recovered is None:
        candidate = _extract_candidate(stripped)
        if candidate is not None:
            recovered = _load_dict(candidate)
    return strict_obj is not None, recovered


def score_prediction(gold: dict, raw: str) -> dict:
    strict, recovered = parse_model_output(raw)
    fields: dict[str, bool] = {}
    for key in KEYS:
        if not isinstance(recovered, dict) or key not in recovered:
            fields[key] = False
            continue
        if key == "amount_usd":
            fields[key] = amounts_equal(recovered[key], gold[key])
        else:
            fields[key] = recovered[key] == gold[key]
    valid = schema_valid(recovered) if recovered is not None else False
    return {
        "strict_json": strict,
        "schema_valid": valid,
        "canonical_exact": raw.strip() == canonical_json(gold),
        "exact_match": bool(valid and all(fields.values())),
        "fields": fields,
        "recovered": recovered,
    }


def aggregate_scores(scored_rows: list[dict]) -> dict:
    n = len(scored_rows)
    if n == 0:
        raise ValueError("cannot aggregate an empty split")

    def count(flag: str) -> int:
        return sum(1 for row in scored_rows if row[flag])

    per_field_correct = {
        key: sum(1 for row in scored_rows if row["fields"][key]) for key in KEYS
    }
    field_slots = n * len(KEYS)
    field_correct_slots = sum(per_field_correct.values())
    return {
        "n": n,
        "strict_json_correct": count("strict_json"),
        "schema_valid_correct": count("schema_valid"),
        "canonical_exact_correct": count("canonical_exact"),
        "exact_match_correct": count("exact_match"),
        "per_field_correct": per_field_correct,
        "field_correct_slots": field_correct_slots,
        "field_slots": field_slots,
    }


def prediction_record(example: dict, raw: str, judgment: dict) -> dict:
    return {
        "id": example["id"],
        "split": example["split"],
        "frame_id": example["frame_id"],
        "gold": example["label"],
        "raw": raw,
        "strict_json": judgment["strict_json"],
        "schema_valid": judgment["schema_valid"],
        "canonical_exact": judgment["canonical_exact"],
        "exact_match": judgment["exact_match"],
        "fields": judgment["fields"],
        "recovered": judgment["recovered"],
    }


def predictions_text(rows: list[dict]) -> str:
    lines = [json.dumps(row, ensure_ascii=True) for row in rows]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_metrics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_predictions(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = predictions_text(rows)
    path.write_text(text, encoding="utf-8")
    return text


def read_predictions(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
