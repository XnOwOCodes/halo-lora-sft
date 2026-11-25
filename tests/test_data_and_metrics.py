"""Locks for the generator and the scorer. These tests do not load a model."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from ticket_sft.baselines import predict_amount, predict_label, predict_order_id, score_split
from ticket_sft.config import DATA_DIR, SEED
from ticket_sft.data import build_dataset, sha256_file, write_dataset
from ticket_sft.metrics import score_prediction
from ticket_sft.prompts import SYSTEM_PROMPT, build_messages
from ticket_sft.schema import INTENTS, KEYS, PRIORITIES, canonical_json


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.gold = {
            "intent": "refund",
            "order_id": None,
            "amount_usd": 12,
            "priority": "low",
        }
        self.raw = canonical_json(self.gold)

    def test_canonical_exact(self):
        scored = score_prediction(self.gold, self.raw)
        self.assertTrue(scored["strict_json"])
        self.assertTrue(scored["schema_valid"])
        self.assertTrue(scored["canonical_exact"])
        self.assertTrue(scored["exact_match"])
        self.assertTrue(all(scored["fields"].values()))

    def test_pretty_json_is_semantic_exact_only(self):
        pretty = json.dumps(json.loads(self.raw), indent=2)
        scored = score_prediction(self.gold, pretty)
        self.assertTrue(scored["strict_json"])
        self.assertTrue(scored["exact_match"])
        self.assertFalse(scored["canonical_exact"])

    def test_fence_and_preamble(self):
        fenced = "```json\n" + self.raw + "\n```"
        scored = score_prediction(self.gold, fenced)
        self.assertFalse(scored["strict_json"])
        self.assertTrue(scored["exact_match"])
        preamble = "Here is the result:\n" + self.raw
        scored = score_prediction(self.gold, preamble)
        self.assertFalse(scored["strict_json"])
        self.assertTrue(scored["exact_match"])

    def test_wrong_intent_and_extra_key(self):
        wrong = json.loads(self.raw)
        wrong["intent"] = "other"
        scored = score_prediction(self.gold, json.dumps(wrong))
        self.assertTrue(scored["schema_valid"])
        self.assertFalse(scored["exact_match"])
        self.assertFalse(scored["fields"]["intent"])
        self.assertTrue(scored["fields"]["priority"])

        extra = json.loads(self.raw)
        extra["note"] = "x"
        scored = score_prediction(self.gold, json.dumps(extra))
        self.assertFalse(scored["schema_valid"])
        self.assertFalse(scored["exact_match"])
        self.assertTrue(scored["fields"]["intent"])

    def test_missing_null_field_is_not_correct(self):
        missing = {"intent": "refund", "amount_usd": 12, "priority": "low"}
        scored = score_prediction(self.gold, json.dumps(missing))
        self.assertFalse(scored["fields"]["order_id"])
        self.assertFalse(scored["schema_valid"])
        self.assertFalse(scored["exact_match"])

    def test_amount_typing(self):
        as_float = json.loads(self.raw)
        as_float["amount_usd"] = 12.0
        scored = score_prediction(self.gold, json.dumps(as_float))
        self.assertTrue(scored["fields"]["amount_usd"])
        self.assertTrue(scored["exact_match"])

        as_text = json.loads(self.raw)
        as_text["amount_usd"] = "12"
        scored = score_prediction(self.gold, json.dumps(as_text))
        self.assertFalse(scored["schema_valid"])
        self.assertFalse(scored["fields"]["amount_usd"])

    def test_unparseable(self):
        scored = score_prediction(self.gold, "I cannot help with that.")
        self.assertFalse(scored["strict_json"])
        self.assertFalse(scored["schema_valid"])
        self.assertFalse(scored["exact_match"])
        self.assertTrue(not any(scored["fields"].values()))

    def test_amount_regex_does_not_read_the_order_id(self):
        text = "Order ORD-12345 costs $40. Later notes mention 15 USD as well."
        self.assertEqual(predict_order_id(text), "ORD-12345")
        self.assertEqual(predict_amount(text), 40)
        self.assertIsNone(predict_amount("See ORD-12345 only."))


class PromptTests(unittest.TestCase):
    def test_prompt_lists_the_schema(self):
        for key in KEYS:
            self.assertIn(key, SYSTEM_PROMPT)
        for intent in INTENTS:
            self.assertIn(intent, SYSTEM_PROMPT)
        for priority in PRIORITIES:
            self.assertIn(priority, SYSTEM_PROMPT)
        self.assertIn("ORD-#####", SYSTEM_PROMPT)

    def test_messages_hide_the_frame(self):
        message = "Please refund ORD-10001. Case 100."
        label = {
            "intent": "refund",
            "order_id": "ORD-10001",
            "amount_usd": None,
            "priority": "medium",
        }
        without = build_messages(message)
        self.assertEqual([item["role"] for item in without], ["system", "user"])
        self.assertEqual(without[1]["content"], "Customer message:\n" + message)
        with_label = build_messages(message, label)
        self.assertEqual(with_label[2]["content"], canonical_json(label))
        self.assertNotIn("tr-refund", with_label[1]["content"])


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = build_dataset()

    def test_counts_and_seed(self):
        self.assertEqual(SEED, 42)
        self.assertEqual(len(self.splits["train"]), 200)
        self.assertEqual(len(self.splits["iid"]), 40)
        self.assertEqual(len(self.splits["ood"]), 40)

    def test_deterministic(self):
        again = build_dataset()
        self.assertEqual(again, self.splits)

    def test_intent_balance_and_disjoint_ids(self):
        train_intents = Counter(row["label"]["intent"] for row in self.splits["train"])
        self.assertEqual(set(train_intents), set(INTENTS))
        self.assertTrue(all(count == 40 for count in train_intents.values()))

        def orders(name):
            return {row["label"]["order_id"] for row in self.splits[name] if row["label"]["order_id"]}

        self.assertTrue(orders("train").isdisjoint(orders("iid")))
        self.assertTrue(orders("train").isdisjoint(orders("ood")))
        self.assertTrue(orders("iid").isdisjoint(orders("ood")))
        messages = [row["message"] for rows in self.splits.values() for row in rows]
        self.assertEqual(len(messages), len(set(messages)))

    def test_heuristic_behavior(self):
        for name in ("train", "iid"):
            metrics, _ = score_split(self.splits[name])
            self.assertEqual(metrics["exact_match_correct"], metrics["n"])
            self.assertEqual(metrics["per_field_correct"]["order_id"], metrics["n"])
            self.assertEqual(metrics["per_field_correct"]["amount_usd"], metrics["n"])

        ood = self.splits["ood"]
        metrics, _ = score_split(ood)
        other = sum(1 for row in ood if row["label"]["intent"] == "other")
        medium = sum(1 for row in ood if row["label"]["priority"] == "medium")
        solvable = sum(
            1
            for row in ood
            if row["label"]["intent"] == "other" and row["label"]["priority"] == "medium"
        )
        self.assertEqual(metrics["per_field_correct"]["intent"], other)
        self.assertEqual(metrics["per_field_correct"]["priority"], medium)
        self.assertEqual(metrics["per_field_correct"]["order_id"], metrics["n"])
        self.assertEqual(metrics["per_field_correct"]["amount_usd"], metrics["n"])
        self.assertEqual(metrics["exact_match_correct"], solvable)
        self.assertGreater(solvable, 0)
        self.assertLess(solvable, len(ood))
        self.assertEqual(metrics["strict_json_correct"], metrics["n"])

    def test_prompt_does_not_include_bookkeeping(self):
        for rows in self.splits.values():
            for row in rows:
                messages = build_messages(row["message"], row["label"])
                self.assertNotIn(row["frame_id"], messages[1]["content"])
                self.assertNotIn(row["id"], messages[1]["content"])
                self.assertEqual(messages[2]["content"], canonical_json(row["label"]))

    def test_files_roundtrip_and_match_committed_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a"
            second = Path(tmp) / "b"
            write_dataset(self.splits, first)
            write_dataset(self.splits, second)
            for name in ("train.jsonl", "eval_iid.jsonl", "eval_ood.jsonl", "manifest.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        if not (DATA_DIR / "train.jsonl").exists():
            return
        with tempfile.TemporaryDirectory() as tmp:
            write_dataset(self.splits, Path(tmp))
            for name in ("train.jsonl", "eval_iid.jsonl", "eval_ood.jsonl", "manifest.json"):
                self.assertEqual(
                    sha256_file(DATA_DIR / name),
                    sha256_file(Path(tmp) / name),
                    msg=f"committed {name} does not match the generator",
                )


if __name__ == "__main__":
    unittest.main()
