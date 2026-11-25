"""Plot the logged training loss. Points are not smoothed."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ticket_sft.config import LOSS_CURVE_PATH, TRAIN_SUMMARY_PATH


def plot_loss(summary_path: Path = TRAIN_SUMMARY_PATH, output_path: Path = LOSS_CURVE_PATH) -> Path:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = summary["loss_history"]
    if len(history) < 2:
        raise SystemExit(f"{summary_path} has too few loss points to plot")
    steps = [point["step"] for point in history]
    losses = [point["loss"] for point in history]
    model = summary["model_id"].split("/")[-1]
    figure, axis = plt.subplots(figsize=(7.0, 4.0), dpi=120)
    axis.plot(steps, losses, color="#1f4e79", linewidth=1.5, marker="o", markersize=3.0)
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Training loss")
    axis.set_title(f"{model} LoRA training loss (seed {summary['seed']})")
    axis.grid(True, linewidth=0.4, alpha=0.4)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def main() -> None:
    path = plot_loss()
    print(path)


if __name__ == "__main__":
    main()
