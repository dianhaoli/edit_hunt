"""Gap hunt (2026-10-01, after the screen found no MDAS-over-DAS gap; user: "look for gap more").
Targets where DAS leaks into the other attributes (low Iso) are where MDAS can beat it:
 A) city_pop Language / Country: DAS and MDAS at full strength (300 steps, 1200 training pairs per prompt attribute)
    at layers 11-15 (handoff ~L15), k in {16, 128}. Shortcuts at the new layers: full, mean-diff (Stage-2 code).
 B) bind name Q at layers 16-19 (binding info reaches the final token at L15+), DAS / MDAS k in {16, 128}, plus
    full / mean-diff at those layers.
All scored on the Stage-2/3 Entity pools (seed 0) and Context pools; selection on Entity, then the seed check
(4 seeds, fresh pools) for anything with ref >= 0.5 and gap >= 0.2 vs best DAS / untrained shortcut.
"""
from __future__ import annotations

import json
import sys
import time

import torch

from . import pointer as P
from .core import Model, iv_add, iv_interchange, iv_set
from .das import train
from .evalkit import SCREEN, build_pool, load_survivor, mean_by_value, metrics, predict_pool, source_reps, train_reps

OUT = SCREEN / "gap_hunt.json"


def city(M, R, log):
    F = load_survivor("city_pop")
    pools = {sp: build_pool(F, sp, 200, seed={"entity": 0, "context": 1}[sp]) for sp in ("entity", "context")}
    layers = [11, 12, 13, 14, 15]
    tr_ents, tr = train_reps(M, F, layers)
    for L in layers:
        src = {sp: source_reps(M, p, [L])[L] for sp, p in pools.items()}
        for a in ("Language",):  # trimmed after 3 runs (~7 min/run): Country dropped
            key = f"city_pop|{a}|L{L}|shortcuts"
            if key not in R:
                mu = mean_by_value(F, tr_ents, tr[L], a)
                z = torch.zeros(M.d)
                res = {}
                for sp, p in pools.items():
                    full = metrics(F, p, predict_pool(M, p, lambda idx: iv_set(L, src[sp][idx])), a)
                    vec = torch.stack([mu.get(F.labels[q.es][a], z) - mu.get(F.labels[q.eb][a], z) for q in p])
                    md = metrics(F, p, predict_pool(M, p, lambda idx: iv_add(L, vec[idx])), a)
                    res[sp] = dict(full=full, meandiff=md)
                R[key] = res
                OUT.write_text(json.dumps(R, indent=1))
            for k, m in ((128, "mdas"), (16, "mdas"), (16, "das")):  # trimmed grid
                if True:
                    key = f"city_pop|{a}|L{L}|k{k}|{m}|full"
                    if key in R:
                        continue
                    U, info = train(M, F, a, L, k, mdas=(m == "mdas"), seed=0, steps=300, n_train=1200, max_seconds=600)
                    res = {sp: metrics(F, p, predict_pool(M, p, lambda idx: iv_interchange(L, U, src[sp][idx])), a)
                           for sp, p in pools.items()}
                    R[key] = dict(info=info, **res)
                    OUT.write_text(json.dumps(R, indent=1))
                    log(f"{key}: E C {res['entity']['cause']:.2f} I {res['entity']['iso']:.2f} S {res['entity']['score']:.2f}"
                        f" | ctx {res['context']['score']:.2f}")


def bind(M, R, log):
    fam = "bind"
    Rp = json.loads((P.OUT / "bind.json").read_text())
    kept_t = [tuple(x) for x in Rp["validity"]["kept"]]
    pairs = {sp: P.make_pairs(M, fam, sp, 200, {"entity": 0, "context": 1}[sp], kept_t) for sp in ("entity", "context")}
    trp = P.make_pairs(M, fam, "train", 600, 1000, kept_t)
    tr_items = [b for v in trp.values() for b, _, _ in v]
    for L in (16, 17, 18, 19):
        srcs = {sp: {kk: P.capture(M, fam, [s for _, s, _ in vv], [L])[L] for kk, vv in p.items()} for sp, p in pairs.items()}
        key = f"bind|Q|L{L}|shortcuts"
        if key not in R:
            reps = P.capture(M, fam, tr_items, [L])[L]
            vals = sorted({it["Q"] for it in tr_items})
            mu = {x: reps[torch.tensor([it["Q"] == x for it in tr_items])].mean(0) for x in vals}
            z = torch.zeros(M.d)
            res = {}
            for sp, p in pairs.items():
                full = P.score_rows(fam, P.run_pairs(M, fam, p, L, lambda k_, ps, src: iv_set(L, src), srcs[sp]))
                md = P.score_rows(fam, P.run_pairs(M, fam, p, L, lambda k_, ps, src: iv_add(
                    L, torch.stack([mu.get(s["Q"], z) - mu.get(b["Q"], z) for b, s, _ in ps])), srcs[sp]))
                res[sp] = dict(full=full, meandiff=md)
            R[key] = res
            OUT.write_text(json.dumps(R, indent=1))
            log(f"{key}: full E {res['entity']['full']['Q']['score']:.2f} P {res['entity']['full']['P']['score']:.2f}")
        tr_src = {kk: P.capture(M, fam, [s for _, s, _ in vv], [L])[L] for kk, vv in trp.items()}
        for k in (16, 128):
            for m in ("das", "mdas"):
                key = f"bind|Q|L{L}|k{k}|{m}"
                if key in R:
                    continue
                ex, srows = [], []
                for kname in ["cause_Q"] + (["iso_Q"] if m == "mdas" else []):
                    for j, (b, s, e) in enumerate(trp[kname]):
                        ex.append((b, e)); srows.append(tr_src[kname][j])
                U, info = P.fit_subspace(M, fam, L, k, ex, torch.stack(srows), steps=300, max_seconds=600)
                res = {sp: P.score_rows(fam, P.run_pairs(M, fam, p, L, lambda k_, ps, src: iv_interchange(L, U, src),
                                                         srcs[sp]))["Q"] for sp, p in pairs.items()}
                R[key] = dict(info=info, **res)
                OUT.write_text(json.dumps(R, indent=1))
                log(f"{key}: E C {res['entity']['cause']:.2f} I {res['entity']['iso']:.2f} S {res['entity']['score']:.2f}"
                    f" | ctx {res['context']['score']:.2f}")


if __name__ == "__main__":
    t0 = time.time()
    log = lambda s: print(f"  {s} ({time.time() - t0:.0f}s)", flush=True)
    R = json.loads(OUT.read_text()) if OUT.exists() else {}
    M = Model()
    for part in sys.argv[1:]:
        {"city": city, "bind": bind}[part](M, R, log)
