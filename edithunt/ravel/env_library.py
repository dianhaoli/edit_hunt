"""Permanent script library for the Stage-5 city environment: every scripted run graded by env.grade, one row each.
Sources: validate.json (reference x4 seeds, full, das k256, sae per instance), recipe_check.json (recipe interface:
reference, das k256/k16/k8/k4/k2/k1, modest_correct), ceiling.json (reference + privileged ceiling per fold).
Scores use the current KL rule (hinge at the fixed budget); runs recorded under the old linear rule are re-scored from
their stored raw Cause / Iso / KL.
Usage: python -m edithunt.ravel.env_library  -> results/screen/env/script_library.json + a printed table
"""
from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean, pstdev

from ..common import ROOT

D = ROOT / "results" / "screen" / "env"


def kl_factor(kl, budget):
    return 1 - min(1.0, max(0.0, kl - budget) / budget)


def rows():
    val = json.loads((D / "validate.json").read_text())
    B = val["kl_budget"]
    out = []
    for k, v in val["runs"].items():
        iid, rec, s = k.split("|")
        g = v["raw"]
        out.append(dict(instance=iid, script=f"{rec}" if rec != "mdas" else "reference", interface="examples",
                        seed=int(s[1:]), cause=g["cause"], iso=g["iso"], kl=g["kl"],
                        score=g["cause"] * g["iso"] * kl_factor(g["kl"], B), source="validate.json"))
    if (D / "recipe_check.json").exists():
        for k, v in json.loads((D / "recipe_check.json").read_text()).items():
            iid, name = k.split("|")
            g = v["grade"]
            out.append(dict(instance=iid, script=name, interface="recipes", seed=0, cause=g["cause"], iso=g["iso"],
                            kl=g["kl"], score=g["cause"] * g["iso"] * kl_factor(g["kl"], B), source="recipe_check.json"))
    if (D / "ceiling.json").exists():
        for k, v in json.loads((D / "ceiling.json").read_text()).items():
            parts = k.split("|")
            if parts[1] == "reference":
                for f, g in enumerate(v["folds"]):
                    out.append(dict(instance=parts[0], script="reference", interface="recipes", seed=0, fold=f,
                                    cause=g["cause"], iso=g["iso"], kl=g["kl"], score=g["score"], soft=g.get("soft_score"),
                                    source="ceiling.json"))
            else:
                g = v["grade"]
                out.append(dict(instance=parts[0], script="CEILING(privileged)", interface="privileged", seed=0,
                                fold=int(parts[2][1:]), cause=g["cause"], iso=g["iso"], kl=g["kl"], score=g["score"],
                                soft=g.get("soft_score"), source="ceiling.json"))
    return out, B


if __name__ == "__main__":
    R, B = rows()
    (D / "script_library.json").write_text(json.dumps(dict(kl_budget=B, rows=R), indent=1))
    by = defaultdict(lambda: defaultdict(list))
    for r in R:
        by[r["instance"]][r["script"]].append(r["score"])
    for iid in sorted(by):
        cells = []
        for sc, v in sorted(by[iid].items(), key=lambda x: -mean(x[1])):
            cells.append(f"{sc} {mean(v):.2f}" + (f"±{pstdev(v):.2f}(n{len(v)})" if len(v) > 1 else ""))
        print(f"{iid}: " + " | ".join(cells))
