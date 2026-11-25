"""Label schema for the support-ticket extraction task."""

from __future__ import annotations

import json
import math
import re

INTENTS = ("refund", "shipping", "billing", "account", "other")
PRIORITIES = ("low", "medium", "high")
KEYS = ("intent", "order_id", "amount_usd", "priority")

ORDER_ID_RE = re.compile(r"ORD-\d{5}")
ORDER_ID_FULL_RE = re.compile(r"^ORD-\d{5}$")


def canonical_json(label: dict) -> str:
    """Compact JSON with a fixed key order. Amounts are integer dollars or null."""
    amount = label["amount_usd"]
    if isinstance(amount, float):
        if not math.isfinite(amount) or not amount.is_integer():
            raise ValueError(f"amount is not an integer number of dollars: {amount}")
        amount = int(amount)
    if amount is not None and not isinstance(amount, int):
        raise TypeError(f"amount must be int or null, got {type(amount).__name__}")
    payload = {
        "intent": label["intent"],
        "order_id": label["order_id"],
        "amount_usd": amount,
        "priority": label["priority"],
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def amounts_equal(predicted, gold) -> bool:
    if predicted is None and gold is None:
        return True
    if isinstance(predicted, bool) or isinstance(gold, bool):
        return False
    if isinstance(predicted, (int, float)) and isinstance(gold, (int, float)):
        if isinstance(predicted, float) and not math.isfinite(predicted):
            return False
        if isinstance(gold, float) and not math.isfinite(gold):
            return False
        return abs(float(predicted) - float(gold)) < 1e-6
    return False


def schema_valid(obj) -> bool:
    """True when obj is a dict with exactly the four keys and legal value types."""
    if not isinstance(obj, dict):
        return False
    if set(obj) != set(KEYS):
        return False
    if obj["intent"] not in INTENTS:
        return False
    if obj["priority"] not in PRIORITIES:
        return False
    order_id = obj["order_id"]
    if order_id is not None and not (
        isinstance(order_id, str) and ORDER_ID_FULL_RE.fullmatch(order_id)
    ):
        return False
    amount = obj["amount_usd"]
    if amount is not None:
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            return False
        if isinstance(amount, float) and not math.isfinite(amount):
            return False
    return True
