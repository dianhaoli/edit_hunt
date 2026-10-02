"""Learnability report for the luna runs: score spread per instance, strategy per run (k, examples, keep recipes),
next to the script library (reference / DAS / full) and the ceiling; regrade of saved submissions under the
hidden-readout grader (GPU).
Usage: python -m edithunt.ravel.env_learn_report [--hidden]  -> results/screen/env/learn_report.json
"""
from __future__ import annotations

import glob
import json
import sys
from statistics import mean, pstdev

import torch

from ..common import ROOT

D = ROOT / "results" / "screen" / "env"


def strategy(rec):
    out = []
    for x in rec["tool_log"]:
        if x["tool"] != "train_subspace":
            continue
        a = x["args"]
        rs = a.get("recipes") or []
        keep = any(str(r.get("answer", "")).startswith("base:") for r in rs) or \
            any(str(e.get("answer")) for e in (a.get("examples") or []) if False)
        n = sum(int(r.get("n", 0)) for r in rs) + len(a.get("examples") or [])
        out.append(dict(k=a.get("k"), steps=a.get("steps"), n=n, keep_recipes=keep))
    return out


if __name__ == "__main__":
    runs = {}
    for p in sorted(glob.glob(str(D / "runs" / "gpt-6-luna" / "Language_s*_r*.json"))):
        r = json.load(open(p))
        runs.setdefault(r["instance"], []).append(r)
    lib = json.loads((D / "script_library.json").read_text())["rows"]
    ceil = json.loads((D / "ceiling.json").read_text())
    rep = {}
    if "--hidden" in sys.argv:
        from . import env as E
        from .core import Model
        from .env_hidden import combined, hidden_iso
        M = Model()
        E.KL_BUDGET = json.loads((D / "validate.json").read_text())["kl_budget"]
        F = E.load_survivor("city_pop")
    for iid, rs in runs.items():
        sc = [r["grade"]["score"] for r in rs]
        ref = [x["score"] for x in lib if x["instance"] == iid and x["script"] == "reference"]
        lazy = max(x["score"] for x in lib if x["instance"] == iid and x["script"] not in ("reference", "modest_correct", "CEILING(privileged)"))
        cfolds = [ceil[f"{iid}|ceiling|f{f}"]["grade"]["score"] for f in (0, 1) if f"{iid}|ceiling|f{f}" in ceil]
        rep[iid] = dict(scores=sc, mean=mean(sc), sd=pstdev(sc), passed=sum(s >= 0.5 for s in sc), n=len(sc),
                        reference_mean=mean(ref), best_lazy=lazy, ceiling_folds=cfolds,
                        cost=sum(r["cost_usd"] for r in rs), strategies=[strategy(r) for r in rs],
                        final_k=[(strategy(r) or [{}])[-1].get("k") for r in rs],
                        used_keep=[any(t["keep_recipes"] for t in strategy(r)) for r in rs])
        if "--hidden" in sys.argv:
            inst = E.load_instance(iid)
            bases = {b for b, _ in inst["grade_pairs"]}
            hs = []
            for r in rs:
                path = r["submission"].get("U_path")
                if not path:
                    hs.append(None); continue
                sub = dict(featurizer=None, layer=inst["layer"], U=torch.load(path), sae_ids=None)
                hs.append(combined(E.grade(M, inst, sub), hidden_iso(M, F, inst, sub, bases), inst["keep"])["score_hidden"])
            rep[iid]["hidden_scores"] = hs
        o = rep[iid]
        print(f"{iid}: luna {[round(s, 2) for s in sc]} mean {o['mean']:.2f} sd {o['sd']:.2f} pass {o['passed']}/{o['n']} | "
              f"reference {o['reference_mean']:.2f} | best lazy {o['best_lazy']:.2f} | ceiling folds {[round(c, 2) for c in cfolds]} | "
              f"final k {o['final_k']} | keep used {sum(o['used_keep'])}/{o['n']} | ${o['cost']:.2f}"
              + (f" | hidden {[round(h, 2) if h is not None else None for h in o['hidden_scores']]}" if 'hidden_scores' in o else ""))
    (D / "learn_report.json").write_text(json.dumps(rep, indent=1))
