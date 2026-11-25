"""Hardware and package versions recorded next to a measured run."""

from __future__ import annotations

import importlib.metadata as metadata
import os
import platform
import subprocess
from pathlib import Path

PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "datasets",
    "accelerate",
    "tokenizers",
    "huggingface_hub",
    "safetensors",
    "numpy",
    "matplotlib",
)


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def cpu_model() -> str | None:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def git_info() -> dict:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    commit = run(["git", "rev-parse", "HEAD"])
    porcelain = run(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "dirty": None if porcelain is None else bool(porcelain),
    }


def runtime_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model(),
        "packages": package_versions(),
        "git": git_info(),
    }
    try:
        import torch
    except ImportError:
        info["torch"] = None
        info["cuda_available"] = None
        info["torch_num_threads"] = None
        return info
    info["torch"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["torch_num_threads"] = int(torch.get_num_threads())
    return info
