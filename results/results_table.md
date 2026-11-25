These figures are computed from `results/*.json` and the prediction jsonl files.
They are not typed in by hand. Exact match is semantic: recovered JSON must be
schema-valid and every field must match the gold label.

Hardware: Intel(R) Xeon(R) Processor, 4 CPUs, Python 3.12.3, torch 2.15.0+cpu. CUDA available: false. Seed 42.

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`. Train rows: 200. Eval rows: iid 40, ood 40. LoRA r=8, alpha=16, dropout=0.05, targets `q_proj, k_proj, v_proj, o_proj`. Trainable parameters: 921,600 / 135,436,608 (0.68%). Supervised tokens: 15.1% of training tokens. Max sequence length seen: 212 (limit 512, truncated 0).

Training loop wall time: 202.5s. Trainer-reported train_runtime: 202.4s. Optimizer steps: 100. Final logged training loss: 0.0945. Minimum logged training loss: 0.0572. Baseline eval wall time: 112.5s. Post-training eval wall time: 59.4s.

Generation is greedy (`do_sample=false`, `max_new_tokens=64`). One item on either 40-row eval split is 2.5 percentage points. The keyword heuristic is not a neural model. It always emits canonical JSON, so its strict-JSON rate is 1 by construction.

In-distribution exact match is higher after LoRA (0/40 to 17/40). Paraphrase exact match is higher after LoRA (0/40 to 2/40).

| Split | System | Strict JSON | Canonical JSON | Schema valid | Exact match | Field accuracy | Intent | Order id | Amount | Priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iid | base SmolLM2-135M-Instruct | 1/40 (2.5%) | 0/40 (0.0%) | 0/40 (0.0%) | 0/40 (0.0%) | 2/160 (1.2%) | 1/40 (2.5%) | 0/40 (0.0%) | 0/40 (0.0%) | 1/40 (2.5%) |
| iid | SmolLM2-135M-Instruct + LoRA | 40/40 (100.0%) | 17/40 (42.5%) | 25/40 (62.5%) | 17/40 (42.5%) | 126/160 (78.8%) | 34/40 (85.0%) | 24/40 (60.0%) | 39/40 (97.5%) | 29/40 (72.5%) |
| iid | keyword heuristic (train cues) | 40/40 (100.0%) | 40/40 (100.0%) | 40/40 (100.0%) | 40/40 (100.0%) | 160/160 (100.0%) | 40/40 (100.0%) | 40/40 (100.0%) | 40/40 (100.0%) | 40/40 (100.0%) |
| ood | base SmolLM2-135M-Instruct | 1/40 (2.5%) | 0/40 (0.0%) | 0/40 (0.0%) | 0/40 (0.0%) | 8/160 (5.0%) | 1/40 (2.5%) | 0/40 (0.0%) | 2/40 (5.0%) | 5/40 (12.5%) |
| ood | SmolLM2-135M-Instruct + LoRA | 40/40 (100.0%) | 2/40 (5.0%) | 14/40 (35.0%) | 2/40 (5.0%) | 98/160 (61.2%) | 18/40 (45.0%) | 18/40 (45.0%) | 39/40 (97.5%) | 23/40 (57.5%) |
| ood | keyword heuristic (train cues) | 40/40 (100.0%) | 4/40 (10.0%) | 40/40 (100.0%) | 4/40 (10.0%) | 108/160 (67.5%) | 8/40 (20.0%) | 40/40 (100.0%) | 40/40 (100.0%) | 20/40 (50.0%) |

Training loss is token cross-entropy on the supervised completion, not the exact-match score.
Each plotted point is one logged optimizer step. The curve is not smoothed.
Loss curve file: `results/loss_curve.png`.

Data sha256: train `64b198e4c60a2907dacd0521dc7700ab02033890aa0788ff73632ec58b16b840`, iid `0abd5beb9d9b1b78bd06fa802786445f9748a73d6d59cf75718835a3871f2352`, ood `b6bb3b7ce842b96faf65d48b32d5c43112553c9227455c144bd8864c7f9b2ab6`.
System prompt sha256 `00726ec182db0c0c33cb773d0e0458ed815bca1a8d952704850bd43f7a3fbd1f`.

In-distribution example (`iid-0000`), first row in the predictions file, not selected by hand:

Gold: `{"intent":"billing","order_id":"ORD-30012","amount_usd":45,"priority":"medium"}`

Base model output:

```
The customer message indicates that they were charged $45 on the order ID 30012, which is the order number. The order ID is 30012, and the customer message states that they were charged $45.
```

Base fields correct: intent=no, order_id=no, amount_usd=no, priority=no.

LoRA output:

```
{"intent":"billing","order_id":"ORD-30012","amount_usd":45,"priority":"medium"}
```

LoRA fields correct: intent=yes, order_id=yes, amount_usd=yes, priority=yes.

Paraphrase example (`ood-0000`), first row in the predictions file, not selected by hand:

Gold: `{"intent":"other","order_id":null,"amount_usd":null,"priority":"medium"}`

Base model output:

```
Here is a JSON object that represents the customer message:

'''json
{
  "intent": "refund",
  "order_id": "1234567890",
  "amount_usd": 1000,
  "priority": "high",
```

Base fields correct: intent=no, order_id=no, amount_usd=no, priority=no.

LoRA output:

```
{"intent":"other","order_id":"634","amount_usd":null,"priority":"medium"}
```

LoRA fields correct: intent=yes, order_id=no, amount_usd=yes, priority=yes.
