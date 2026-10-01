"""Stage 3: DAS (Cause loss) and MDAS (Cause + Iso loss), k in {16, 128}, the 3 Stage-2 layers, Entity split, seed 0.
Evaluated on the same Entity pool as Stage 2 (200 items per prompt attribute, seed 0).
Output: results/screen/stage3/{family}.json (checkpoint per run); MDAS subspaces in ~/screen_subspaces/.
Usage: python -m edithunt.ravel.stage3 [family ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from .core import Model, iv_interchange
from .das import train
from .evalkit import SCREEN, build_pool, load_survivor, metrics, predict_pool, source_reps
from .stage2 import N_EVAL

OUT = SCREEN / "stage3"
SUBS = Path.home() / "screen_subspaces"


def evaluate(M, F, pool, src, L, U, target):
    preds = predict_pool(M, pool, lambda idx: iv_interchange(L, U, src[idx]))
    return metrics(F, pool, preds, target)


def run_family(M: Model, name: str, seeds=(0,), splits=("entity",), targets=None, ks=(16, 128),
               methods=("das", "mdas"), layers=None, out_name=None):
    t0 = time.time()
    F = load_survivor(name)
    s2 = json.loads((SCREEN / "stage2" / f"{name}.json").read_text())
    layers = layers or s2["layers"]
    path = OUT / f"{out_name or name}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    SUBS.mkdir(exist_ok=True)
    R = json.loads(path.read_text()) if path.exists() else {"family": name, "layers": layers, "runs": {}}
    pools = {sp: build_pool(F, sp, N_EVAL, seed={"entity": 0, "context": 1}[sp]) for sp in splits}
    for L in layers:
        srcs = None
        for a in targets or F.targets:
            for k in ks:
                for m in methods:
                    for seed in seeds:
                        key = f"{a}|L{L}|k{k}|{m}|s{seed}"
                        if key in R["runs"]:
                            continue
                        if srcs is None:
                            srcs = {sp: source_reps(M, p, [L])[L] for sp, p in pools.items()}
                        U, info = train(M, F, a, L, k, mdas=(m == "mdas"), seed=seed)
                        if m == "mdas":
                            torch.save(U, SUBS / f"{name}_{a.replace(' ', '_')}_L{L}_k{k}_{m}_s{seed}.pt")
                        res = {sp: evaluate(M, F, pools[sp], srcs[sp], L, U, a) for sp in pools}
                        R["runs"][key] = dict(info=info, **res)
                        path.write_text(json.dumps(R, indent=1))
                        e = res[splits[0]]
                        print(f"  {name} {key}: cause {e['cause']:.2f} iso {e['iso']:.2f} score {e['score']:.2f} "
                              f"loss {info['loss_first']:.2f}->{info['loss_last']:.2f} {info['seconds']:.0f}s "
                              f"(total {time.time() - t0:.0f}s)", flush=True)
    return R


def main(names):
    M = Model()
    for n in names:
        run_family(M, n)


if __name__ == "__main__":
    main(sys.argv[1:])
