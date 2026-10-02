"""Noise next to the pass margin (CPU): bootstrap over private base cities of the per-item records in ceiling.json
(reference full grade), hard and soft scores; compared with seed SD from the script library.
Usage: python -m edithunt.ravel.env_noise  -> results/screen/env/noise.json
"""
from __future__ import annotations

import json
import random
from statistics import mean, pstdev

from ..common import ROOT
from . import env as E

D = ROOT / "results" / "screen" / "env"

if __name__ == "__main__":
    C = json.loads((D / "ceiling.json").read_text())
    lib = json.loads((D / "script_library.json").read_text())["rows"]
    out = {}
    for key, v in C.items():
        if not key.endswith("|reference"):
            continue
        iid = key.split("|")[0]
        inst = E.load_instance(iid)
        g = v["full"]
        items, keep, f = g["items"], inst["keep"], g["kl_factor"]
        bases = sorted({r[1] for r in items})
        rng = random.Random(0)
        boot = []
        for _ in range(1000):
            bs = [rng.choice(bases) for _ in bases]
            sel = [r for b in bs for r in items if r[1] == b]  # with multiplicity
            boot.append(E.metrics_from_items(sel, keep, f))
        seeds = [r["score"] for r in lib if r["instance"] == iid and r["script"] == "reference" and r["source"] == "validate.json"]
        out[iid] = dict(score=g["score"], soft=g["soft_score"],
                        boot_sd=pstdev([b["score"] for b in boot]), boot_sd_soft=pstdev([b["soft_score"] for b in boot]),
                        boot_p05=sorted(b["score"] for b in boot)[50], seed_sd=pstdev(seeds) if len(seeds) > 1 else None,
                        seed_mean=mean(seeds) if seeds else None, margin_over_pass=g["score"] - E.PASS)
        o = out[iid]
        print(f"{iid}: score {o['score']:.2f} (margin {o['margin_over_pass']:+.2f}) | bootstrap SD {o['boot_sd']:.3f} "
              f"(5th pct {o['boot_p05']:.2f}) | seed SD {o['seed_sd'] or 0:.3f} | soft {o['soft']:.2f} SD {o['boot_sd_soft']:.3f}")
    (D / "noise.json").write_text(json.dumps(out, indent=1))
