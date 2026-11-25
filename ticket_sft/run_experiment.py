"""Run data generation, both evals, LoRA training, the plot, and the results check.

Times and commands recorded here are wall-clock measurements of this process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from ticket_sft.config import (
    BASELINE_METRICS_PATH,
    KEYWORD_METRICS_PATH,
    POST_METRICS_PATH,
    ROOT,
    RUN_META_PATH,
    TRAIN_SUMMARY_PATH,
)
from ticket_sft.runtime_info import git_info, runtime_info

ENV_OVERRIDES = {
    "PYTHONHASHSEED": "42",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TQDM_DISABLE": "1",
    "PYTHONUNBUFFERED": "1",
}

STEPS = (
    ("data", ["ticket_sft.data"]),
    ("tests", ["unittest", "tests.test_data_and_metrics", "-v"]),
    ("keyword", ["ticket_sft.baselines"]),
    ("baseline_eval", ["ticket_sft.evaluate", "--which", "base"]),
    ("train", ["ticket_sft.train"]),
    ("post_eval", ["ticket_sft.evaluate", "--which", "adapter"]),
    ("plot", ["ticket_sft.plot_loss"]),
    ("report", ["ticket_sft.report"]),
)


def _exact(path, split: str) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    block = payload["splits"][split]
    return [block["exact_match_correct"], block["n"]]


def main() -> None:
    env = os.environ.copy()
    env.update(ENV_OVERRIDES)
    stages = {}
    argv = []
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    for name, args in STEPS:
        command = [sys.executable, "-m", *args]
        argv.append(command)
        print(f"=== {name}: {' '.join(command)} ===", flush=True)
        step_clock = time.perf_counter()
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        stages[name] = time.perf_counter() - step_clock
        print(f"=== {name} done in {stages[name]:.1f}s ===", flush=True)

    summary = json.loads(TRAIN_SUMMARY_PATH.read_text(encoding="utf-8"))
    meta = {
        "status": "measured",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - clock,
        "stages_seconds": stages,
        "argv": argv,
        "env": ENV_OVERRIDES,
        "train_wall_seconds": summary["train_wall_seconds"],
        "trainer_train_runtime_seconds": summary["trainer_train_runtime_seconds"],
        "exact_match": {
            "iid_base": _exact(BASELINE_METRICS_PATH, "iid"),
            "ood_base": _exact(BASELINE_METRICS_PATH, "ood"),
            "iid_adapter": _exact(POST_METRICS_PATH, "iid"),
            "ood_adapter": _exact(POST_METRICS_PATH, "ood"),
            "iid_keyword": _exact(KEYWORD_METRICS_PATH, "iid"),
            "ood_keyword": _exact(KEYWORD_METRICS_PATH, "ood"),
        },
        "git": git_info(),
        "runtime": runtime_info(),
    }
    RUN_META_PATH.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"=== check: {sys.executable} -m ticket_sft.check_results ===", flush=True)
    subprocess.run(
        [sys.executable, "-m", "ticket_sft.check_results"],
        cwd=ROOT,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
