"""T2_keepstate v2 (RAVEL-style) validation, no agent spend. Stages (resumable; rows appended to JSONL):

  shortcut  step 4: at L in SHORTCUT_LAYERS, every candidate pair: null edit, plain mean-diff, naive gradient
            (target capital only) capped at |mean-diff|, naive gradient uncapped, random (|mean-diff|).
  oracle    steps 5-6: at L in LAYERS: offline oracle (4 seeds) and DAS rank-1 (seed 0), plus the step-4 methods
            at layers not covered there.
  tools     step 7: tool-path reference (ref_keepstate through the agent tools) on the oracle-validated instances;
            saves the validated instances.
  report    step 8: method x layer table -> results/suite/t2_ravel/summary.json

  PYTHONPATH=. .venv/bin/python experiments/t2_ravel.py --stage shortcut [--shard 0/2]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from edithunt.common import ROOT
from edithunt.model import Subject
from edithunt.env import ravel as R

OUT = ROOT / "results" / "suite" / "t2_ravel"
SHORTCUT_LAYERS = [6, 8, 12, 16, 20]
STEPS, LR = 150, 0.05
SEEDS = [0, 1, 2, 3]
PASS, STRICT = 0.5, 0.75
KEYS = ("Cause", "Iso_state", "Iso_other")


def row(g: dict, **kw) -> dict:
    r = {**kw, "reward": g["reward"], "reward_product": g["reward_product"], "kl_mean": g["kl_mean"],
         "Iso": g["Iso"], "Disentangle": g["Disentangle"], "continuous": g["continuous"]}
    for k in KEYS:
        r[k], r[k + "_n"] = g[k]["rate"], g[k]["n"]
    r["state_parts"] = {k: v["rate"] for k, v in g["Iso_state_parts"].items()}
    return r


def done_keys(p: Path) -> set:
    if not p.exists():
        return set()
    return {(r["pair"], r["layer"], r["method"], r.get("seed", 0)) for r in map(json.loads, p.read_text().splitlines())}


def pairs_data(S, shard):
    pairs = R.candidate_pairs()
    k, n = shard
    out = []
    for i, pr in enumerate(pairs):
        if i % n != k:
            continue
        P = R.make_pair(S, *pr)
        print(pr, "->", "ok" if P else "too few cities", flush=True)
        if P:
            out.append(P)
    return out


def baselines(S, inst, L, pair, todo, emit, seed=0):
    md = R.meandiff(S, inst, L)
    nrm = md.norm().item()
    meth = {"null": lambda: [(L, "city_last", md * 0)],
            "meandiff": lambda: [(L, "city_last", md)],
            "random": lambda: R.random_vec(S, nrm, L, seed),
            "naive_capped": lambda: R.train_vector(S, inst, L, seed, STEPS, keep=False, max_norm=nrm, lr=LR),
            "naive_uncapped": lambda: R.train_vector(S, inst, L, seed, STEPS, keep=False, max_norm=None, lr=LR)}
    for m, f in meth.items():
        if (pair, L, m, seed) in todo:
            continue
        t = time.time()
        e = f()
        g = R.grade_ravel(S, inst, None, edits=e)
        emit(row(g, pair=pair, layer=L, method=m, seed=seed, md_norm=nrm,
                 norm=e[0][2].norm().item(), secs=time.time() - t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["shortcut", "oracle", "tools", "report"])
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    shard = tuple(map(int, a.shard.split("/")))
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "report":
        return report()
    S = Subject("Qwen/Qwen2.5-1.5B")
    p = OUT / f"{a.stage}.{shard[0]}.jsonl"
    todo = done_keys(p)
    f = p.open("a")

    def emit(r):
        f.write(json.dumps(r) + "\n"); f.flush()
        print(r["pair"], r["layer"], r["method"], r.get("seed", 0),
              " ".join(f"{k}={r[k]:.2f}" for k in KEYS), f"kl={r['kl_mean']:.3f} R={r['reward']:.3f}", flush=True)

    for P in pairs_data(S, shard):
        pair = f"{P['source']}->{P['target']}"
        if a.stage == "shortcut":
            for L in SHORTCUT_LAYERS:
                baselines(S, R.make_instance(S, P, L), L, pair, todo, emit)
        elif a.stage == "oracle":
            for L in R.LAYERS[S.name]:
                inst = R.make_instance(S, P, L)
                if L not in SHORTCUT_LAYERS:
                    baselines(S, inst, L, pair, todo, emit)
                nrm = R.meandiff(S, inst, L).norm().item()
                for sd in SEEDS:
                    if (pair, L, "oracle", sd) not in todo:
                        t = time.time()
                        e = R.train_vector(S, inst, L, sd, STEPS, keep=True, max_norm=nrm, lr=LR)
                        emit(row(R.grade_ravel(S, inst, None, edits=e), pair=pair, layer=L, method="oracle", seed=sd,
                                 md_norm=nrm, norm=e[0][2].norm().item(), secs=time.time() - t))
                if (pair, L, "das", 0) not in todo:
                    t = time.time()
                    e = R.train_vector(S, inst, L, 0, STEPS, keep=True, das=True, lr=LR)
                    emit(row(R.grade_ravel(S, inst, None, edits=e), pair=pair, layer=L, method="das", seed=0,
                             md_norm=nrm, secs=time.time() - t))
        elif a.stage == "tools":
            from edithunt.env.instance import save_instance
            from edithunt.env.tasks import run_reference
            valid = validated()
            for L in R.LAYERS[S.name]:
                if (pair, L) not in valid:
                    continue
                inst = R.make_instance(S, P, L)
                inst.extra["oracle"] = valid[(pair, L)]
                save_instance(inst, ROOT / "results" / "suite" / "instances" / "T2_keepstate_v2")
                if (pair, L, "tool_ref", 0) in todo:
                    continue
                t = time.time()
                r = run_reference(S, inst, seed=0)
                g = r["grade"]
                if not g.get("valid"):
                    print(pair, L, "tool_ref invalid", g.get("reason"), r["error"], flush=True)
                    continue
                emit(row(g, pair=pair, layer=L, method="tool_ref", seed=0, error=r["error"],
                         budget_used=r["budget_used"], secs=time.time() - t))


def load(stage: str) -> list[dict]:
    return [json.loads(l) for p in sorted(OUT.glob(f"{stage}.*.jsonl")) for l in p.read_text().splitlines()]


def validated(thr: float = PASS) -> dict:
    """(pair, layer) -> oracle seed rewards, for instances whose oracle passes on >= 3 of 4 seeds."""
    by = {}
    for r in load("oracle"):
        if r["method"] == "oracle":
            by.setdefault((r["pair"], r["layer"]), []).append(r["reward"])
    return {k: v for k, v in by.items() if len(v) == len(SEEDS) and sum(x >= thr for x in v) >= 3}


def report():
    import statistics as st
    rows = load("shortcut") + load("oracle") + load("tools")
    agg = lambda rs, k: st.mean(r[k] for r in rs) if rs else None

    def table(rs, thr_list=(PASS, STRICT)):
        out = {}
        for m in sorted({r["method"] for r in rs}):
            for L in sorted({r["layer"] for r in rs if r["method"] == m}):
                x = [r for r in rs if r["method"] == m and r["layer"] == L]
                out[f"{m}@L{L}"] = {"n": len(x), **{f"pass@{t}": sum(r["reward"] >= t for r in x) / len(x) for t in thr_list},
                                    **{k: agg(x, k) for k in ("reward", "reward_product", *KEYS, "kl_mean")}}
        return out

    oracle_by = {}
    for r in rows:
        if r["method"] == "oracle":
            oracle_by.setdefault((r["pair"], r["layer"]), []).append(r["reward"])
    validity = {}
    for thr in (PASS, STRICT):
        per = {}
        for (pair, L), v in oracle_by.items():
            if len(v) == len(SEEDS):
                per.setdefault(L, []).append(sum(x >= thr for x in v) >= 3)
        validity[f"thr{thr}"] = {L: {"n": len(v), "valid": sum(v), "rejected": len(v) - sum(v)} for L, v in sorted(per.items())}
    V = set(validated())
    summary = {"shortcut": table([r for r in rows if r["layer"] in SHORTCUT_LAYERS and
                                  r["method"] in ("null", "meandiff", "random", "naive_capped", "naive_uncapped")]),
               "all_candidates": table(rows),
               "validated": table([r for r in rows if (r["pair"], r["layer"]) in V]),
               "instance_validity": validity, "n_validated": len(V)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
