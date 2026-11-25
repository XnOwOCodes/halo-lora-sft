"""Deterministic support-ticket dataset.

Labels come from the frame that produced the message, not from a human annotator
and not from the keyword heuristic. The heuristic is checked against those labels
so a leaked training cue in a paraphrase frame fails before any model is trained.

Train and in-distribution eval share sentence frames. They do not share order ids,
dollar amounts draws, or case numbers. The paraphrase (ood) split uses different
frames and different cue words for intent and priority.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from ticket_sft.baselines import predict_label
from ticket_sft.config import (
    AMOUNTS,
    DATA_DIR,
    DATA_VERSION,
    IID_REPEATS,
    OOD_REPEATS,
    SEED,
    TRAIN_REPEATS,
)
from ticket_sft.schema import INTENTS, KEYS, PRIORITIES, amounts_equal

CASE_RE = re.compile(r"Case (\d+)\.$")


@dataclass(frozen=True)
class Frame:
    id: str
    family: str
    intent: str
    priority: str
    presence: str
    text: str


def _frame(frame_id: str, family: str, intent: str, priority: str, presence: str, text: str) -> Frame:
    return Frame(frame_id, family, intent, priority, presence, text)


FRAMES: tuple[Frame, ...] = (
    # Training frames. Intent and priority cues here are the heuristic's whole lexicon.
    _frame("tr-refund-hb", "train", "refund", "high", "both",
           "Please refund {order_id}. The total was ${amount}. This is urgent."),
    _frame("tr-refund-ho", "train", "refund", "high", "order",
           "I need a refund on {order_id} right away."),
    _frame("tr-refund-ha", "train", "refund", "high", "amount",
           "I want my money back for the ${amount} I paid, as soon as possible."),
    _frame("tr-refund-lb", "train", "refund", "low", "both",
           "Can I get my money back for {order_id}? The total was {amount} USD. No rush."),
    _frame("tr-refund-la", "train", "refund", "low", "amount",
           "I would like my money back, {amount} USD. Whenever you can."),
    _frame("tr-refund-mb", "train", "refund", "medium", "both",
           "I want to refund {order_id}. The total was ${amount}."),
    _frame("tr-refund-mn", "train", "refund", "medium", "neither",
           "Hi, I am looking for a refund."),
    _frame("tr-refund-mo", "train", "refund", "medium", "order",
           "Please refund {order_id}."),
    _frame("tr-ship-hb", "train", "shipping", "high", "both",
           "Where is my package for {order_id}? I paid ${amount}. This is urgent."),
    _frame("tr-ship-ho", "train", "shipping", "high", "order",
           "The delivery for {order_id} has not arrived. I need this right away."),
    _frame("tr-ship-ha", "train", "shipping", "high", "amount",
           "Please look at the shipping for the ${amount} purchase as soon as possible."),
    _frame("tr-ship-lb", "train", "shipping", "low", "both",
           "Tracking on {order_id} looks stuck. It was {amount} USD. No rush."),
    _frame("tr-ship-la", "train", "shipping", "low", "amount",
           "Has the package shipped? It was {amount} USD. Whenever you can."),
    _frame("tr-ship-mb", "train", "shipping", "medium", "both",
           "I have a question about shipping for {order_id}. The total was ${amount}."),
    _frame("tr-ship-mn", "train", "shipping", "medium", "neither",
           "Just checking on a delivery."),
    _frame("tr-ship-mo", "train", "shipping", "medium", "order",
           "Shipping status for {order_id}."),
    _frame("tr-bill-hb", "train", "billing", "high", "both",
           "You charged me ${amount} on {order_id}. This is urgent."),
    _frame("tr-bill-ho", "train", "billing", "high", "order",
           "There is a wrong invoice on {order_id}. I need this right away."),
    _frame("tr-bill-ha", "train", "billing", "high", "amount",
           "The invoice for ${amount} looks wrong. As soon as possible."),
    _frame("tr-bill-lb", "train", "billing", "low", "both",
           "Please look at the billing for {order_id}. It was {amount} USD. No rush."),
    _frame("tr-bill-la", "train", "billing", "low", "amount",
           "I have an invoice question about {amount} USD. Whenever you can."),
    _frame("tr-bill-mb", "train", "billing", "medium", "both",
           "Why was I charged ${amount} on {order_id}?"),
    _frame("tr-bill-mn", "train", "billing", "medium", "neither",
           "I have a billing question."),
    _frame("tr-bill-mo", "train", "billing", "medium", "order",
           "Billing issue on {order_id}."),
    _frame("tr-acct-hb", "train", "account", "high", "both",
           "I cannot log in, and the account for {order_id} shows ${amount}. This is urgent."),
    _frame("tr-acct-ho", "train", "account", "high", "order",
           "My account is locked for {order_id}. Please help right away."),
    _frame("tr-acct-ha", "train", "account", "high", "amount",
           "I forgot my password. The last total was ${amount}. As soon as possible."),
    _frame("tr-acct-lb", "train", "account", "low", "both",
           "I cannot access my account for {order_id}. Balance {amount} USD. No rush."),
    _frame("tr-acct-la", "train", "account", "low", "amount",
           "Password help for a balance of {amount} USD. Whenever you can."),
    _frame("tr-acct-mb", "train", "account", "medium", "both",
           "Account help for {order_id}. Listed amount ${amount}."),
    _frame("tr-acct-mn", "train", "account", "medium", "neither",
           "I forgot my password."),
    _frame("tr-acct-mo", "train", "account", "medium", "order",
           "Login problem on {order_id}."),
    _frame("tr-other-hb", "train", "other", "high", "both",
           "I have a question about {order_id}. I paid ${amount}. This is urgent."),
    _frame("tr-other-ho", "train", "other", "high", "order",
           "Something else about {order_id}. I need this right away."),
    _frame("tr-other-ha", "train", "other", "high", "amount",
           "Please call me about a ${amount} issue as soon as possible."),
    _frame("tr-other-lb", "train", "other", "low", "both",
           "General question on {order_id}. Total {amount} USD. No rush."),
    _frame("tr-other-la", "train", "other", "low", "amount",
           "Not sure who to ask about {amount} USD. Whenever you can."),
    _frame("tr-other-mb", "train", "other", "medium", "both",
           "I have a miscellaneous question about {order_id} for ${amount}."),
    _frame("tr-other-mn", "train", "other", "medium", "neither",
           "Just a general question."),
    _frame("tr-other-mo", "train", "other", "medium", "order",
           "Other request for {order_id}."),
    # Paraphrase frames. These must not be solvable by the training-cue heuristic,
    # except for the default classes (intent other, priority medium) and the
    # order-id / amount regular expressions.
    _frame("od-refund-hb", "ood", "refund", "high", "both",
           "Please reimburse me for {order_id}. The payment was USD {amount} and I need this immediately."),
    _frame("od-refund-lo", "ood", "refund", "low", "order",
           "Could the payment be returned for {order_id}? Take your time."),
    _frame("od-refund-ma", "ood", "refund", "medium", "amount",
           "I am asking to be reimbursed {amount} dollars."),
    _frame("od-refund-mn", "ood", "refund", "medium", "neither",
           "Hello, I would like to be reimbursed."),
    _frame("od-ship-hb", "ood", "shipping", "high", "both",
           "The parcel for {order_id} never showed up. I paid USD {amount} and this can't wait."),
    _frame("od-ship-lo", "ood", "shipping", "low", "order",
           "Any news on the courier drop-off for {order_id}? No hurry."),
    _frame("od-ship-ma", "ood", "shipping", "medium", "amount",
           "I am writing about a courier drop-off that cost {amount} dollars."),
    _frame("od-ship-mn", "ood", "shipping", "medium", "neither",
           "Wondering about my parcel."),
    _frame("od-bill-hb", "ood", "billing", "high", "both",
           "My card was billed USD {amount} under {order_id} and this can't wait."),
    _frame("od-bill-lo", "ood", "billing", "low", "order",
           "Could you review a duplicate card payment on {order_id}? Take your time."),
    _frame("od-bill-ma", "ood", "billing", "medium", "amount",
           "I am disputing {amount} dollars on my card."),
    _frame("od-bill-mn", "ood", "billing", "medium", "neither",
           "Question about a payment on my card."),
    _frame("od-acct-hb", "ood", "account", "high", "both",
           "I am locked out of my sign-in for {order_id}. Last payment USD {amount}. This can't wait."),
    _frame("od-acct-lo", "ood", "account", "low", "order",
           "Could you restore access for {order_id}? No hurry."),
    _frame("od-acct-ma", "ood", "account", "medium", "amount",
           "I cannot sign in. The balance was {amount} dollars."),
    _frame("od-acct-mn", "ood", "account", "medium", "neither",
           "Help with signing in."),
    _frame("od-other-hb", "ood", "other", "high", "both",
           "This does not fit your usual topics. Reference {order_id}, paid USD {amount}, and this can't wait."),
    _frame("od-other-lo", "ood", "other", "low", "order",
           "No specific department, about {order_id}. Take your time."),
    _frame("od-other-ma", "ood", "other", "medium", "amount",
           "A general note regarding {amount} dollars."),
    _frame("od-other-mn", "ood", "other", "medium", "neither",
           "I have an unusual request."),
)


def _fail(message: str) -> None:
    raise ValueError(message)


def _placeholders_ok(frame: Frame) -> None:
    has_order = "{order_id}" in frame.text
    has_amount = "{amount}" in frame.text
    if has_order != (frame.presence in {"both", "order"}):
        _fail(f"{frame.id} order placeholder does not match presence {frame.presence}")
    if has_amount != (frame.presence in {"both", "amount"}):
        _fail(f"{frame.id} amount placeholder does not match presence {frame.presence}")
    leftover = frame.text.replace("{order_id}", "").replace("{amount}", "")
    if "{" in leftover or "}" in leftover:
        _fail(f"{frame.id} has an unexpected placeholder")


def render_message(frame: Frame, order_id: str, amount: int, case_no: int) -> str:
    kwargs = {}
    if "{order_id}" in frame.text:
        kwargs["order_id"] = order_id
    if "{amount}" in frame.text:
        kwargs["amount"] = amount
    body = frame.text.format(**kwargs)
    return f"{body} Case {case_no}."


def make_label(frame: Frame, order_id: str, amount: int) -> dict:
    return {
        "intent": frame.intent,
        "order_id": order_id if "{order_id}" in frame.text else None,
        "amount_usd": amount if "{amount}" in frame.text else None,
        "priority": frame.priority,
    }


def _labels_match(predicted: dict, gold: dict) -> bool:
    for key in KEYS:
        if key == "amount_usd":
            if not amounts_equal(predicted[key], gold[key]):
                return False
        elif predicted[key] != gold[key]:
            return False
    return True


def validate_frames(frames: tuple[Frame, ...] = FRAMES) -> None:
    seen = set()
    for frame in frames:
        if frame.id in seen:
            _fail(f"duplicate frame id {frame.id}")
        seen.add(frame.id)
        if frame.family not in {"train", "ood"}:
            _fail(f"{frame.id} has family {frame.family}")
        if frame.intent not in INTENTS:
            _fail(f"{frame.id} has intent {frame.intent}")
        if frame.priority not in PRIORITIES:
            _fail(f"{frame.id} has priority {frame.priority}")
        if frame.presence not in {"both", "order", "amount", "neither"}:
            _fail(f"{frame.id} has presence {frame.presence}")
        if "ORD-" in frame.text or "Case " in frame.text:
            _fail(f"{frame.id} hardcodes an id")
        if not (frame.text.endswith(".") or frame.text.endswith("?")):
            _fail(f"{frame.id} does not end with punctuation")
        _placeholders_ok(frame)
        message = render_message(frame, "ORD-11111", 13, 999)
        gold = make_label(frame, "ORD-11111", 13)
        predicted = predict_label(message)
        if not amounts_equal(predicted["amount_usd"], gold["amount_usd"]):
            _fail(f"{frame.id} amount regex got {predicted['amount_usd']} from {message}")
        if predicted["order_id"] != gold["order_id"]:
            _fail(f"{frame.id} order regex got {predicted['order_id']} from {message}")
        if frame.family == "train":
            if not _labels_match(predicted, gold):
                _fail(
                    f"{frame.id} train cue mismatch: predicted {predicted} gold {gold} text {message}"
                )
            continue
        if frame.intent == "other":
            if predicted["intent"] != "other":
                _fail(f"{frame.id} paraphrase leaked a train intent cue: {message}")
        elif predicted["intent"] == frame.intent:
            _fail(f"{frame.id} paraphrase is solved by train intent cues: {message}")
        if frame.priority == "medium":
            if predicted["priority"] != "medium":
                _fail(f"{frame.id} paraphrase medium frame contains a train priority cue: {message}")
        elif predicted["priority"] == frame.priority:
            _fail(f"{frame.id} paraphrase priority is solved by train cues: {message}")


def _shuffled_range(rng: random.Random, start: int, count: int) -> list[int]:
    values = list(range(start, start + count))
    rng.shuffle(values)
    return values


def _generate_split(name: str, family: str, repeats: int, seed_offset: int, order_start: int, case_start: int) -> list[dict]:
    frames = [frame for frame in FRAMES if frame.family == family]
    if not frames:
        _fail(f"no frames for family {family}")
    rng = random.Random(SEED + seed_offset)
    sequence: list[Frame] = []
    for _ in range(repeats):
        sequence.extend(frames)
    rng.shuffle(sequence)
    count = len(sequence)
    order_pool = _shuffled_range(rng, order_start, count)
    case_pool = _shuffled_range(rng, case_start, count)
    rows = []
    for index, frame in enumerate(sequence):
        order_value = order_pool[index]
        if not 10000 <= order_value <= 99999:
            _fail(f"order id {order_value} is not five digits")
        order_id = f"ORD-{order_value}"
        amount = rng.choice(AMOUNTS)
        case_no = case_pool[index]
        message = render_message(frame, order_id, amount, case_no)
        label = make_label(frame, order_id, amount)
        rows.append(
            {
                "id": f"{name}-{index:04d}",
                "split": name,
                "frame_id": frame.id,
                "presence": frame.presence,
                "message": message,
                "label": label,
            }
        )
    return rows


def build_dataset() -> dict[str, list[dict]]:
    validate_frames()
    splits = {
        "train": _generate_split("train", "train", TRAIN_REPEATS, 0, 10000, 100),
        "iid": _generate_split("iid", "train", IID_REPEATS, 1, 30000, 400),
        "ood": _generate_split("ood", "ood", OOD_REPEATS, 2, 60000, 600),
    }
    audit(splits)
    return splits


def audit(splits: dict[str, list[dict]]) -> None:
    messages = []
    case_numbers = []
    order_ids = {name: set() for name in splits}
    for name, rows in splits.items():
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            _fail(f"duplicate ids in {name}")
        for row in rows:
            message = row["message"]
            messages.append(message)
            label = row["label"]
            predicted = predict_label(message)
            if predicted["order_id"] != label["order_id"]:
                _fail(f"{row['id']} order id mismatch: {message}")
            if not amounts_equal(predicted["amount_usd"], label["amount_usd"]):
                _fail(f"{row['id']} amount mismatch: {message}")
            if label["order_id"] is None and "ORD-" in message:
                _fail(f"{row['id']} has a stray order id")
            if label["order_id"] is not None and label["order_id"] not in message:
                _fail(f"{row['id']} order id is missing from the message")
            case_match = CASE_RE.search(message)
            if case_match is None:
                _fail(f"{row['id']} is missing a case number")
            case_numbers.append(int(case_match.group(1)))
            if label["order_id"] is not None:
                order_ids[name].add(label["order_id"])
            if name in {"train", "iid"}:
                if not _labels_match(predicted, label):
                    _fail(f"{row['id']} heuristic does not match the gold label")
            else:
                if label["intent"] == "other":
                    if predicted["intent"] != "other":
                        _fail(f"{row['id']} other paraphrase matched a train cue")
                elif predicted["intent"] == label["intent"]:
                    _fail(f"{row['id']} paraphrase intent is solved by train cues")
                if label["priority"] == "medium":
                    if predicted["priority"] != "medium":
                        _fail(f"{row['id']} medium paraphrase contains a priority cue")
                elif predicted["priority"] == label["priority"]:
                    _fail(f"{row['id']} paraphrase priority is solved by train cues")
            if row["frame_id"] in message:
                _fail(f"{row['id']} leaked frame id {row['frame_id']} into the message")
    if len(messages) != len(set(messages)):
        _fail("duplicate messages across the dataset")
    if len(case_numbers) != len(set(case_numbers)):
        _fail("duplicate case numbers")
    if not order_ids["train"].isdisjoint(order_ids["iid"]):
        _fail("train and iid share an order id")
    if not order_ids["train"].isdisjoint(order_ids["ood"]):
        _fail("train and ood share an order id")
    if not order_ids["iid"].isdisjoint(order_ids["ood"]):
        _fail("iid and ood share an order id")
    train_frames = {row["frame_id"] for row in splits["train"]}
    iid_frames = {row["frame_id"] for row in splits["iid"]}
    ood_frames = {row["frame_id"] for row in splits["ood"]}
    if iid_frames != train_frames:
        _fail("iid does not cover the training frames")
    if not ood_frames.isdisjoint(train_frames):
        _fail("ood reuses a training frame")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=True) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            if "id" not in row or "message" not in row or "label" not in row:
                _fail(f"{path}:{line_number} is missing id, message, or label")
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_dataset(splits: dict[str, list[dict]], directory: Path | None = None) -> dict:
    directory = DATA_DIR if directory is None else directory
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        "train": directory / "train.jsonl",
        "iid": directory / "eval_iid.jsonl",
        "ood": directory / "eval_ood.jsonl",
    }
    for name, path in files.items():
        write_jsonl(path, splits[name])
    manifest = {
        "data_version": DATA_VERSION,
        "seed": SEED,
        "split_seeds": {"train": SEED, "iid": SEED + 1, "ood": SEED + 2},
        "counts": {name: len(rows) for name, rows in splits.items()},
        "sha256": {path.name: sha256_file(path) for path in files.values()},
        "files": {name: path.name for name, path in files.items()},
        "order_id_ranges": {"train": [10000, 10000 + len(splits["train"]) - 1],
                            "iid": [30000, 30000 + len(splits["iid"]) - 1],
                            "ood": [60000, 60000 + len(splits["ood"]) - 1]},
        "case_number_ranges": {"train": [100, 100 + len(splits["train"]) - 1],
                               "iid": [400, 400 + len(splits["iid"]) - 1],
                               "ood": [600, 600 + len(splits["ood"]) - 1]},
        "frame_ids": {
            "train": sorted({row["frame_id"] for row in splits["train"]}),
            "ood": sorted({row["frame_id"] for row in splits["ood"]}),
        },
        "label_source": (
            "Labels are assigned by the generator from the frame's intent, priority, "
            "and presence, plus the order id and amount written into that message. "
            "They are not human annotations."
        ),
        "split_definition": {
            "train": "Training frames, repeated.",
            "iid": "Same frames as train, disjoint order ids, amounts, and case numbers.",
            "ood": "Paraphrase frames that do not appear in training.",
        },
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    splits = build_dataset()
    manifest = write_dataset(splits, DATA_DIR)
    print(json.dumps({"counts": manifest["counts"], "sha256": manifest["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
