"""Prompt text shared by training, evaluation, and the unrun Modal recipe."""

from __future__ import annotations

import hashlib

from ticket_sft.schema import INTENTS, KEYS, PRIORITIES, canonical_json

SYSTEM_PROMPT = (
    "Extract support fields from the customer message. "
    "Reply with only a JSON object using keys intent, order_id, amount_usd, and priority. "
    "Do not use markdown.\n"
    "intent: refund (money returned), shipping (delivery problem), "
    "billing (charge or invoice problem), account (cannot sign in), or other.\n"
    "order_id: the ORD-##### id in the message, or null.\n"
    "amount_usd: the dollar amount in the message as a number, or null.\n"
    "priority: high if they want immediate action, low if they say there is no time pressure, "
    "otherwise medium."
)


def system_prompt_sha256() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def user_text(message: str) -> str:
    return f"Customer message:\n{message}"


def build_messages(message: str, label: dict | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text(message)},
    ]
    if label is not None:
        messages.append({"role": "assistant", "content": canonical_json(label)})
    return messages


def render_prompt(tokenizer, message: str) -> str:
    """User-turn prompt, including the assistant header, with no gold answer."""
    return tokenizer.apply_chat_template(
        build_messages(message),
        tokenize=False,
        add_generation_prompt=True,
    )


def render_training_text(tokenizer, message: str, label: dict) -> str:
    """Full conversation, including the gold assistant JSON."""
    return tokenizer.apply_chat_template(
        build_messages(message, label),
        tokenize=False,
        add_generation_prompt=False,
    )


def assert_prompt_lists_schema() -> None:
    for key in KEYS:
        if key not in SYSTEM_PROMPT:
            raise ValueError(f"system prompt is missing key {key}")
    for intent in INTENTS:
        if intent not in SYSTEM_PROMPT:
            raise ValueError(f"system prompt is missing intent {intent}")
    for priority in PRIORITIES:
        if priority not in SYSTEM_PROMPT:
            raise ValueError(f"system prompt is missing priority {priority}")
    if "ORD-#####" not in SYSTEM_PROMPT:
        raise ValueError("system prompt is missing the order-id pattern")
    if "null" not in SYSTEM_PROMPT:
        raise ValueError("system prompt is missing null")
