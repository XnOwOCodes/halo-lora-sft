"""Train-cue keyword heuristic.

This is not a trained model. Intent and priority use only phrases that appear
in the training frames. Order ids and dollar amounts use regular expressions
that cover every format in the train, in-distribution, and paraphrase sets.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ticket_sft.metrics import aggregate_scores, prediction_record, score_prediction
from ticket_sft.schema import ORDER_ID_RE, canonical_json

# Precedence is the tuple order. "other" is the fallback when nothing matches.
INTENT_CUES = (
    ("refund", ("refund", "money back")),
    ("billing", ("charged", "invoice", "billing")),
    ("shipping", ("package", "delivery", "tracking", "shipped", "shipping")),
    ("account", ("account", "password", "log in", "login")),
)

HIGH_CUES = ("as soon as possible", "right away", "urgent")
LOW_CUES = ("whenever you can", "no rush")

AMOUNT_REGEXES = (
    re.compile(r"\$(\d+)"),
    re.compile(r"\b(\d+)\s+USD\b"),
    re.compile(r"\bUSD\s+(\d+)\b"),
    re.compile(r"\b(\d+)\s+dollars\b"),
)


def predict_intent(message: str) -> str:
    text = message.lower()
    for intent, cues in INTENT_CUES:
        if any(cue in text for cue in cues):
            return intent
    return "other"


def predict_priority(message: str) -> str:
    text = message.lower()
    if any(cue in text for cue in HIGH_CUES):
        return "high"
    if any(cue in text for cue in LOW_CUES):
        return "low"
    return "medium"


def predict_order_id(message: str) -> str | None:
    match = ORDER_ID_RE.search(message)
    if match is None:
        return None
    return match.group(0)


def predict_amount(message: str) -> int | None:
    found: list[tuple[int, int]] = []
    for regex in AMOUNT_REGEXES:
        for match in regex.finditer(message):
            found.append((match.start(), int(match.group(1))))
    if not found:
        return None
    found.sort(key=lambda item: item[0])
    return found[0][1]


def predict_label(message: str) -> dict:
    return {
        "intent": predict_intent(message),
        "order_id": predict_order_id(message),
        "amount_usd": predict_amount(message),
        "priority": predict_priority(message),
    }


def heuristic_json(message: str) -> str:
    return canonical_json(predict_label(message))


def score_split(examples: list[dict]) -> tuple[dict, list[dict]]:
    scored = []
    predictions = []
    for example in examples:
        raw = heuristic_json(example["message"])
        judgment = score_prediction(example["label"], raw)
        scored.append(judgment)
        predictions.append(prediction_record(example, raw, judgment))
    return aggregate_scores(scored), predictions


def main() -> None:
    import logging

    from ticket_sft.config import (
        DATA_DIR,
        KEYWORD_METRICS_PATH,
        KEYWORD_PREDICTIONS_PATH,
        LOG_DIR,
        SEED,
    )
    from ticket_sft.data import read_jsonl
    from ticket_sft.logging_utils import setup_logging
    import time
    from datetime import datetime, timezone

    from ticket_sft.metrics import sha256_text, write_metrics, write_predictions

    parser = argparse.ArgumentParser(description="Score the train-cue keyword heuristic.")
    parser.add_argument("--metrics", type=Path, default=KEYWORD_METRICS_PATH)
    parser.add_argument("--predictions", type=Path, default=KEYWORD_PREDICTIONS_PATH)
    parser.add_argument("--log", type=Path, default=LOG_DIR / "keyword_baseline.log")
    args = parser.parse_args()
    setup_logging(args.log)

    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    train = read_jsonl(DATA_DIR / "train.jsonl")
    train_metrics, _ = score_split(train)
    if train_metrics["exact_match_correct"] != train_metrics["n"]:
        raise SystemExit(
            "keyword heuristic missed a training example; the cue lists drifted from the frames"
        )
    logging.info(
        "train sanity check passed: keyword heuristic exact match %s/%s",
        train_metrics["exact_match_correct"],
        train_metrics["n"],
    )

    splits = {
        "iid": read_jsonl(DATA_DIR / "eval_iid.jsonl"),
        "ood": read_jsonl(DATA_DIR / "eval_ood.jsonl"),
    }
    split_metrics = {}
    rows = []
    for name, examples in splits.items():
        metrics, predictions = score_split(examples)
        split_metrics[name] = metrics
        rows.extend(predictions)
        logging.info(
            "%s exact match %s/%s",
            name,
            metrics["exact_match_correct"],
            metrics["n"],
        )

    payload = {
        "status": "measured",
        "which": "keyword_baseline",
        "model_id": None,
        "adapter_path": None,
        "seed": SEED,
        "description": (
            "Deterministic heuristic. Intent and priority use training-frame cues only. "
            "Order id and amount use regular expressions. The heuristic always emits "
            "canonical JSON, so its JSON rate is 1 by construction."
        ),
        "generation": None,
        "splits": split_metrics,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - clock,
    }
    text = write_predictions(args.predictions, rows)
    payload["predictions_sha256"] = sha256_text(text)
    write_metrics(args.metrics, payload)
    logging.info("wrote %s and %s", args.metrics, args.predictions)


if __name__ == "__main__":
    main()
