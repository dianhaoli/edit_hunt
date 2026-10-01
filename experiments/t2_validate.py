"""Step 3 of the T2 v2 follow-up: ONE fixed gated-oracle recipe on 20 pairs x layers {8,12,16,20} x 3 seeds, plus
the shortcuts (null, mean-diff, random, naive gradient capped/uncapped) on the same instances. Grader v2 with the
fixed country term. Instance valid = oracle product reward (Cause * Iso_state * Iso_other * KL factor) >= 0.6 on
>= 2 of 3 seeds at its given layer. Rows streamed to results/suite/t2_ravel/validate.<shard>.jsonl; edits saved.

  PYTHONPATH=. .venv/bin/python experiments/t2_validate.py --recipe gated --shard 0/2
  PYTHONPATH=. .venv/bin/python experiments/t2_validate.py --report
"""
import argparse
import json
import random
import statistics as st
import time

from edithunt.common import ROOT
from edithunt.data import CITIES_EXTRA, study_states
from edithunt.model import Subject
from edithunt.env import ravel as R

D = ROOT / "results" / "suite" / "t2_ravel"
LAYERS, SEEDS, N_PAIRS = [8, 12, 16, 20], [0, 1, 2], 20
# primary bar 0.6 (fixed 2026-10-01 before step-3 oracle results): above every shortcut run seen (max 0.34);
# 0.5 and 0.7 reported for sensitivity
PASS, BARS = 0.6, (0.5, 0.6, 0.7)
RECIPES = {"gated": dict(), "gated_boost": dict(steps=300, templates=["fs1", "zs1", "fs2", "zs2"])}
SHORTCUTS = ["null", "meandiff", "random", "naive_capped", "naive_uncapped"]
KEYS = ("Cause", "Iso_state", "Iso_other")


def pair_list(S):
    """20 valid (source, target) pairs: each extended-list source with a random target, then second targets."""
    tg = study_states(min_cities=5)
    srcs = sorted(s for s in tg if s in CITIES_EXTRA)
    out, skipped = [], []
    for rnd in range(3):
        rng = random.Random(f"t2v2-pairs-{rnd}")
        for s in srcs:
            if len(out) >= N_PAIRS:
                return out, skipped
            t = rng.choice([x for x in tg if x != s and (s, x) not in out])
            (out if R.make_pair(S, s, t) else skipped).append((s, t))
    return out, skipped


def row(g, **kw):
    r = {**kw, **{k: g[k]["rate"] for k in KEYS}, **{k + "_n": g[k]["n"] for k in KEYS},
         "kl_mean": g["kl_mean"], "kl_country": g["kl_country_fixed"]["mean"], "kl_generic": g["kl_generic"]["mean"],
         "kl_mean_oldprobe": g["kl_mean_oldprobe"], "reward_mult": g["reward_mult"], "reward": g["reward"],
         "reward_product": g["reward_product"], "continuous": g["continuous"]}
    return r


def run(recipe, shard):
    k, n = shard
    S = Subject("Qwen/Qwen2.5-1.5B")
    pairs, skipped = pair_list(S)
    print("pairs", pairs, "skipped (too few cities)", skipped, flush=True)
    (D / "edits_v").mkdir(exist_ok=True)
    out = D / f"validate.{'sc' if recipe is None else 'or'}{k}.jsonl"  # shortcuts-only / oracle-only jobs
    done = {(r["pair"], r["layer"], r["method"], r["seed"]) for r in map(json.loads, out.read_text().splitlines())} \
        if out.exists() else set()
    f = out.open("a")

    def emit(inst, L, pair, method, seed, e, t):
        g = R.grade_ravel(S, inst, None, edits=e)
        R.save_edits(D / "edits_v" / f"{pair.replace(' ', '_').replace('->', '_to_')}-L{L}-{method}-s{seed}.pt", e)
        r = row(g, pair=pair, layer=L, method=method, seed=seed, recipe=recipe, secs=time.time() - t)
        f.write(json.dumps(r) + "\n"); f.flush()
        print(pair, L, method, seed, " ".join(f"{x}={r[x]:.2f}" for x in (*KEYS, "kl_mean", "reward_mult")), flush=True)

    for i, (s, t) in enumerate(pairs):
        if i % n != k:
            continue
        pair = f"{s}->{t}"
        P = R.make_pair(S, s, t)
        for L in LAYERS:
            inst = R.make_instance(S, P, L)
            md = R.meandiff(S, inst, L); nrm = md.norm().item()
            sc = {"null": lambda: [(L, "city_last", md * 0)], "meandiff": lambda: [(L, "city_last", md)],
                  "random": lambda: R.random_vec(S, nrm, L, 0),
                  "naive_capped": lambda: R.train_vector(S, inst, L, 0, 150, keep=False, max_norm=nrm),
                  "naive_uncapped": lambda: R.train_vector(S, inst, L, 0, 150, keep=False, max_norm=None)}
            for m in (SHORTCUTS if recipe is None else []):  # oracle jobs skip shortcuts (separate job)
                if (pair, L, m, 0) not in done:
                    t0 = time.time(); emit(inst, L, pair, m, 0, sc[m](), t0)
            for sd in ([] if recipe is None else SEEDS):  # --shortcuts_only: recipe None
                if (pair, L, "oracle", sd) not in done:
                    kw = dict(RECIPES[recipe])
                    t0 = time.time()
                    e = R.train_gated(S, inst, L, sd, kw.pop("steps", 150), keep_sample=True, **kw)
                    emit(inst, L, pair, "oracle", sd, e, t0)


def report():
    rows = [json.loads(l) for p in sorted(D.glob("validate.*.jsonl")) for l in p.read_text().splitlines()]
    orc = {}
    for r in rows:
        if r["method"] == "oracle":
            orc.setdefault((r["pair"], r["layer"]), []).append(r)
    full = {k: v for k, v in orc.items() if len(v) == len(SEEDS)}
    sens = {str(b): sum(sum(x["reward_mult"] >= b for x in v) >= 2 for v in full.values()) for b in BARS}
    inst_ok = {k: sum(x["reward_mult"] >= PASS for x in v) >= 2 for k, v in full.items()}
    per_layer = {L: {"n": sum(1 for (p, l) in inst_ok if l == L), "valid": sum(v for (p, l), v in inst_ok.items() if l == L)}
                 for L in LAYERS}
    V = {k for k, v in inst_ok.items() if v}

    def tab(subset):
        t = {}
        for m in ["oracle"] + SHORTCUTS:
            for L in LAYERS:
                x = [r for r in rows if r["method"] == m and r["layer"] == L and (r["pair"], L) in subset]
                if x:
                    t[f"{m}@L{L}"] = {"n": len(x), "pass": sum(r["reward_mult"] >= PASS for r in x) / len(x),
                                      **{k: st.mean(r[k] for r in x) for k in ("reward_mult", "reward", *KEYS, "kl_mean")}}
        return t
    summ = {"bar": PASS, "valid_by_bar": sens, "n_instances": len(inst_ok), "n_valid": len(V), "valid_fraction": len(V) / max(1, len(inst_ok)),
            "per_layer": per_layer, "valid_instances": sorted(map(list, V)),
            "on_valid": tab(V), "on_all": tab(set(inst_ok))}
    (D / "validate_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: summ[k] for k in ("bar", "valid_by_bar", "n_instances", "n_valid", "valid_fraction", "per_layer")}, indent=1))
    for name in ("on_valid", "on_all"):
        print(f"\n{name}: method@layer  n  pass  R_mult  Cause  IsoS  IsoO  KL")
        for k, v in summ[name].items():
            print(f"  {k:22s} {v['n']:3d} {v['pass']:.2f} {v['reward_mult']:.2f} {v['Cause']:.2f} {v['Iso_state']:.2f} "
                  f"{v['Iso_other']:.2f} {v['kl_mean']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=list(RECIPES))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--shortcuts_only", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(None if a.shortcuts_only else a.recipe, tuple(map(int, a.shard.split("/"))))
