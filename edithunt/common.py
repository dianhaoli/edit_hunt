"""Shared experiment utilities: args, result IO, valid-instance loading."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def base_args(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32)
    return ap


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def rdir(model: str) -> Path:
    d = ROOT / "results" / model.split("/")[-1]
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(model: str, name: str, config: dict, result: dict) -> Path:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = None
    p = rdir(model) / f"{name}.json"
    p.write_text(json.dumps({"config": config, "git": commit, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "result": result}, indent=1, default=float))
    return p


def load(model: str, name: str) -> dict:
    return json.loads((rdir(model) / f"{name}.json").read_text())


def valid_table(model: str) -> dict[str, dict[str, bool]]:
    """template -> city -> valid"""
    return load(model, "phase0_validity")["result"]["valid"]
