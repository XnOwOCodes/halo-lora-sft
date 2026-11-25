"""GPU scale-up of the same LoRA recipe. This file was not executed.

The numbers in README.md and results/ come from the CPU SmolLM2-135M-Instruct
run only. Nothing in this file has been launched, and it does not write into
results/.

The recipe matches the CPU run: same seed, same JSONL files, same prompt, same
LoRA rank, same learning rate, same epoch count, same greedy eval. Two settings
change because the model is larger and the device is a GPU:

- model: Qwen/Qwen3-4B-Instruct-2507
- dtype bfloat16, device cuda, gradient checkpointing on

Launching it creates a paid Modal A100 job. Do not run it unless you mean to.

    pip install modal
    modal setup
    MODAL_TRAIN_I_WILL_PAY=1 modal run modal_train.py

Halo is a separate scale-up path (Hopper or Blackwell images). It is not
invoked here. See the README.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

NOT_RUN_MESSAGE = """\
This Modal recipe has not been run.
The measured results in this repository are the CPU SmolLM2-135M-Instruct run in README.md.

To launch a paid A100 training job (this spends money and is not part of the published results):

    pip install modal
    modal setup
    MODAL_TRAIN_I_WILL_PAY=1 modal run modal_train.py
"""

ROOT = Path(__file__).resolve().parent


def _requirement_pins() -> list[str]:
    pins = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("--"):
            continue
        name = stripped.split("==", 1)[0].split("[", 1)[0].strip().lower()
        if name in {"torch", "torchvision", "torchaudio"}:
            continue
        pins.append(stripped)
    return pins


try:
    import modal
except ImportError:
    modal = None


if modal is None:
    if __name__ == "__main__":
        print(NOT_RUN_MESSAGE)
        raise SystemExit(2)
else:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch", *_requirement_pins())
        .add_local_python_source("ticket_sft")
        .add_local_dir(str(ROOT / "data"), remote_path="/root/data")
    )
    app = modal.App("ticket-slot-lora")

    @app.function(image=image, gpu="A100", timeout=3 * 60 * 60, memory=32768)
    def run_recipe() -> dict:
        from dataclasses import replace

        from ticket_sft.config import (
            EVAL_IID_PATH,
            EVAL_OOD_PATH,
            MAX_NEW_TOKENS,
            MODAL_MODEL_ID,
            SEED,
            TrainConfig,
        )
        from ticket_sft.evaluate import run_evaluation
        from ticket_sft.train import run_training

        config = replace(
            TrainConfig(),
            model_id=MODAL_MODEL_ID,
            dtype="bfloat16",
            device="cuda",
            gradient_checkpointing=True,
        )
        split_files = {"iid": EVAL_IID_PATH, "ood": EVAL_OOD_PATH}
        baseline = run_evaluation(
            model_id=MODAL_MODEL_ID,
            split_files=split_files,
            adapter_path=None,
            dtype="bfloat16",
            device="cuda",
            max_new_tokens=MAX_NEW_TOKENS,
            seed=SEED,
        )
        summary = run_training(config)
        post = run_evaluation(
            model_id=MODAL_MODEL_ID,
            split_files=split_files,
            adapter_path=config.output_dir,
            dtype="bfloat16",
            device="cuda",
            max_new_tokens=MAX_NEW_TOKENS,
            seed=SEED,
        )
        return {
            "published": False,
            "note": "Not a result checked into the repository. The CPU run is the measured one.",
            "baseline_metrics": baseline["metrics"],
            "train_summary": summary,
            "post_metrics": post["metrics"],
        }

    @app.local_entrypoint()
    def main() -> None:
        if os.environ.get("MODAL_TRAIN_I_WILL_PAY") != "1":
            print(NOT_RUN_MESSAGE)
            return
        import json

        print(json.dumps(run_recipe.remote(), indent=2, default=str))


if __name__ == "__main__" and modal is not None:
    print(NOT_RUN_MESSAGE)
    raise SystemExit(2)
