"""Stage 4: rank (family, target attribute) and re-check the top candidates over seeds.

Per (family, target), Entity split, score = Cause x Iso:
  ref   = best MDAS score over the 3 layers x k in {16, 128} (Stage 3, seed 0)
  short = best score among full, meandiff, sae5, sae50, random16 (Stage 2) and DAS (Stage 3)
  GAP   = ref - short;  good = ref >= 0.5 and GAP >= 0.2
  inseparable = every method (incl. MDAS) < 0.5
Seeds: the top ~8 by GAP (good ones first) re-run MDAS at their best (layer, k) with seeds 0-3, evaluated on FRESH
pools (entity seed 2, context seed 3) so the selection pool does not score its own winner. Kept if >= 3/4 seeds
reach score >= 0.5 on the Entity split (Context reported).
Usage: python -m edithunt.ravel.stage4 rank | seeds | plot
"""
from __future__ import annotations

import json
import sys

from .evalkit import SCREEN

SHORT = ("full", "meandiff", "sae5", "sae50", "random16")
N_TOP = 8


def table():
    rows = []
    for p in sorted((SCREEN / "stage3").glob("*.json")):
        if "_seeds" in p.stem:
            continue
        s3 = json.loads(p.read_text())
        fam = s3["family"]
        s2 = json.loads((SCREEN / "stage2" / f"{fam}.json").read_text())
        for a in s2["targets"]:
            best = {}
            for L, res in s2["results"]["entity"].items():
                for m in SHORT:
                    if a in res.get(m, {}):
                        r = res[m][a]
                        if r["score"] == r["score"] and r["score"] > best.get(m, {"score": -1})["score"]:
                            best[m] = dict(r, layer=int(L))
            for key, run in s3["runs"].items():
                ta, L, k, m, s = key.split("|")
                if ta != a:
                    continue
                r = run["entity"]
                if r["score"] == r["score"] and r["score"] > best.get(m, {"score": -1})["score"]:
                    best[m] = dict(r, layer=int(L[1:]), k=int(k[1:]))
            if "mdas" not in best:
                continue
            sm = max((m for m in best if m != "mdas"), key=lambda m: best[m]["score"])
            ref, short = best["mdas"]["score"], best[sm]["score"]
            rows.append(dict(family=fam, target=a, ref=ref, ref_cfg=dict(layer=best["mdas"]["layer"], k=best["mdas"]["k"]),
                             ref_cause=best["mdas"]["cause"], ref_iso=best["mdas"]["iso"],
                             ref_disentangle=best["mdas"]["disentangle"], short=short, short_method=sm,
                             short_cause=best[sm]["cause"], short_iso=best[sm]["iso"], gap=ref - short,
                             good=ref >= 0.5 and ref - short >= 0.2,
                             inseparable=max(b["score"] for b in best.values()) < 0.5,
                             best_by_method={m: dict(score=b["score"], cause=b["cause"], iso=b["iso"],
                                                     disentangle=b["disentangle"], layer=b["layer"], k=b.get("k"))
                                             for m, b in best.items()}))
    from .pointer import table_rows
    for fam in ("bind", "mcqa"):
        rows += table_rows(fam)
    rows.sort(key=lambda r: (-r["good"], -r["gap"]))
    return rows


def rank():
    rows = table()
    seeds_path = SCREEN / "stage4_seeds.json"
    seeds = json.loads(seeds_path.read_text()) if seeds_path.exists() else {}
    for r in rows:
        s = seeds.get(f"{r['family']}|{r['target']}")
        if s:
            ent = [x["entity"]["score"] for x in s["runs"]]
            ctx = [x["context"]["score"] for x in s["runs"]]
            r["seeds"] = dict(entity=ent, context=ctx, pass_entity=sum(v >= 0.5 for v in ent),
                              pass_context=sum(v >= 0.5 for v in ctx), cfg=s["cfg"])
            r["final"] = r["good"] and r["seeds"]["pass_entity"] >= 3
    (SCREEN / "ranking.json").write_text(json.dumps(rows, indent=1))
    print(f"{'family':16s} {'target':17s} {'ref':>5s} {'short':>5s} {'method':9s} {'gap':>5s} good  seeds(E/C)")
    for r in rows:
        sd = f"{r['seeds']['pass_entity']}/4 {r['seeds']['pass_context']}/4" if "seeds" in r else ""
        print(f"{r['family']:16s} {r['target']:17s} {r['ref']:5.2f} {r['short']:5.2f} {r['short_method']:9s} "
              f"{r['gap']:5.2f} {'Y' if r['good'] else '-':4s}{' INSEP' if r['inseparable'] else ''} {sd}")
    return rows


def seeds():
    from .core import Model
    from .das import train
    from .evalkit import build_pool, load_survivor, metrics, predict_pool, source_reps
    from .core import iv_interchange
    from .stage2 import N_EVAL
    rows = table()
    top = [r for r in rows if r["ref"] >= 0.5][:N_TOP]
    path = SCREEN / "stage4_seeds.json"
    S = json.loads(path.read_text()) if path.exists() else {}
    M = Model()
    from .pointer import VARS, seed_runs
    for r in top:
        key = f"{r['family']}|{r['target']}"
        if r["family"] in VARS:
            if key not in S or len(S[key]["runs"]) < 4:
                L, k = r["ref_cfg"]["layer"], r["ref_cfg"]["k"]
                S[key] = {"cfg": dict(layer=L, k=k), "runs": seed_runs(M, r["family"], r["target"], L, k)}
                path.write_text(json.dumps(S, indent=1))
            continue
        F = load_survivor(r["family"])
        L, k = r["ref_cfg"]["layer"], r["ref_cfg"]["k"]
        ent = S.setdefault(key, {"cfg": dict(layer=L, k=k), "runs": []})
        pools = {sp: build_pool(F, sp, N_EVAL, seed={"entity": 2, "context": 3}[sp]) for sp in ("entity", "context")}
        src = {sp: source_reps(M, p, [L])[L] for sp, p in pools.items()}
        for seed in range(len(ent["runs"]), 4):
            U, info = train(M, F, r["target"], L, k, mdas=True, seed=seed)
            res = {sp: metrics(F, pools[sp], predict_pool(M, pools[sp], lambda idx: iv_interchange(L, U, src[sp][idx])),
                               r["target"]) for sp in pools}
            ent["runs"].append(dict(seed=seed, info=info, **res))
            path.write_text(json.dumps(S, indent=1))
            print(f"  {key} L{L} k{k} seed {seed}: entity {res['entity']['score']:.2f} "
                  f"context {res['context']['score']:.2f}", flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.loads((SCREEN / "ranking.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 6))
    import numpy as np
    xs = np.linspace(0, 0.8, 50)
    ax.fill_between(xs, np.maximum(0.5, xs + 0.2), 1.02, color="#d9f0d3", zorder=0, label="good: ref ≥ 0.5, gap ≥ 0.2")
    cols = {"city_pop": "C0", "nobel": "C3", "verb": "C4", "us_cities": "C5", "bind": "C2", "mcqa": "C1"}
    for r in rows:
        mk = "*" if r.get("final") else ("o" if r["good"] else "x")
        ax.scatter(r["short"], r["ref"], c=cols.get(r["family"], "k"), marker=mk, s=110 if mk == "*" else 45, zorder=3)
        ax.annotate(f"{r['family']}:{r['target']}", (r["short"], r["ref"]), fontsize=7, xytext=(4, 3),
                    textcoords="offset points")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    for f, c in cols.items():
        if any(r["family"] == f for r in rows):
            ax.scatter([], [], c=c, label=f)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("best shortcut score (full / mean-diff / SAE / DAS / random), Cause×Iso")
    ax.set_ylabel("reference score (best MDAS), Cause×Iso")
    ax.set_title("Breadth screen, gemma-2-2b, Entity split\n(* = passes ≥3/4 seeds, o = good on seed 0, x = not good)")
    ax.legend(fontsize=7, loc="lower right")
    out = SCREEN.parent / "figures" / "fig5_breadth_screen.png"
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    {"rank": rank, "seeds": seeds, "plot": plot}[sys.argv[1]]()
