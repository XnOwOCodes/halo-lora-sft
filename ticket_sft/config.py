"""Fixed paths and hyperparameters for the published CPU run.

The Modal script builds a separate config for Qwen3-4B. It does not change
these defaults, and it was not executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SEED = 42
DATA_VERSION = 1

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODAL_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

MAX_NEW_TOKENS = 64
MAX_LENGTH = 512
EPOCHS = 2.0
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
LOGGING_STEPS = 1
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

TRAIN_REPEATS = 5
IID_REPEATS = 1
OOD_REPEATS = 2

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
LOG_DIR = RESULTS_DIR / "logs"
ARTIFACTS_DIR = ROOT / "artifacts"
ADAPTER_DIR = ARTIFACTS_DIR / "adapter"
TRAINER_DIR = ARTIFACTS_DIR / "trainer"
README_PATH = ROOT / "README.md"

TRAIN_PATH = DATA_DIR / "train.jsonl"
EVAL_IID_PATH = DATA_DIR / "eval_iid.jsonl"
EVAL_OOD_PATH = DATA_DIR / "eval_ood.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.json"

BASELINE_METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"
BASELINE_PREDICTIONS_PATH = RESULTS_DIR / "baseline_predictions.jsonl"
POST_METRICS_PATH = RESULTS_DIR / "post_metrics.json"
POST_PREDICTIONS_PATH = RESULTS_DIR / "post_predictions.jsonl"
KEYWORD_METRICS_PATH = RESULTS_DIR / "keyword_baseline_metrics.json"
KEYWORD_PREDICTIONS_PATH = RESULTS_DIR / "keyword_baseline_predictions.jsonl"
TRAIN_SUMMARY_PATH = RESULTS_DIR / "train_summary.json"
LOSS_CURVE_PATH = RESULTS_DIR / "loss_curve.png"
RESULTS_TABLE_PATH = RESULTS_DIR / "results_table.md"
RUN_META_PATH = RESULTS_DIR / "run_meta.json"

RESULTS_START = "<!-- RESULTS:START -->"
RESULTS_END = "<!-- RESULTS:END -->"

AMOUNTS = (8, 12, 15, 19, 24, 27, 33, 40, 45, 52, 60, 75, 80, 96, 110, 125)


@dataclass(frozen=True)
class TrainConfig:
    model_id: str = MODEL_ID
    train_file: Path = TRAIN_PATH
    output_dir: Path = ADAPTER_DIR
    trainer_dir: Path = TRAINER_DIR
    summary_path: Path = TRAIN_SUMMARY_PATH
    log_path: Path = LOG_DIR / "train.log"
    seed: int = SEED
    epochs: float = EPOCHS
    learning_rate: float = LEARNING_RATE
    per_device_train_batch_size: int = BATCH_SIZE
    gradient_accumulation_steps: int = GRAD_ACCUM
    max_length: int = MAX_LENGTH
    lora_r: int = LORA_R
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = LORA_DROPOUT
    target_modules: tuple[str, ...] = TARGET_MODULES
    weight_decay: float = WEIGHT_DECAY
    warmup_ratio: float = WARMUP_RATIO
    logging_steps: int = LOGGING_STEPS
    dtype: str = "float32"
    device: str = "cpu"
    gradient_checkpointing: bool = False
    # None trains for the configured number of epochs. The published run stays None.
    max_steps_limit: int | None = None
