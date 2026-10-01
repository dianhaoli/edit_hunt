"""Reference boost (2026-10-01, user: "make the reference solver a little better"): city_pop Language @L12 and
Continent @L10 (layer fixed per task). Grid: k in {128, 256} x keep-weight in {1, 2}, 600 steps, 2400 training pairs per
prompt attribute; DAS gets the same budget (k in {128, 256}) so the gap is like-for-like. Selection on the Stage-2/3
Entity pools (seed 0); then the winner per target is seed-checked on fresh pools (entity 2 / context 3), 4 seeds,
alongside the untrained shortcuts at that layer.
"""
from __future__ import annotations

import json
import sys

import torch

from .core import Model, iv_add, iv_interchange, iv_set
from .das import train
from .evalkit import SCREEN, build_pool, load_survivor, mean_by_value, metrics, predict_pool, source_reps, train_reps
from .sae import SAE, iv_sae, probe_select

OUT = SCREEN / "boost.json"
TASKS = {"Language": 12, "Continent": 10}
BUDGET = dict(steps=600, n_train=2400, max_seconds=900)


def ev(M, F, A, L, pools, src, mk):
    return {sp: metrics(F, p, predict_pool(M, p, lambda idx: mk(sp, idx)), A) for sp, p in pools.items()}


if __name__ == "__main__":
    M = Model()
    F = load_survivor("city_pop")
    R = json.loads(OUT.read_text()) if OUT.exists() else {}
    save = lambda: OUT.write_text(json.dumps(R, indent=1))
    for A, L in TASKS.items():
        sel = {sp: build_pool(F, sp, 200, seed={"entity": 0, "context": 1}[sp]) for sp in ("entity", "context")}
        ssrc = {sp: source_reps(M, p, [L])[L] for sp, p in sel.items()}
        grid = [("mdas", k, w) for k in (128, 256) for w in (1.0, 2.0)] + [("das", k, 1.0) for k in (128, 256)]
        for m, k, w in grid:
            key = f"{A}|L{L}|{m}|k{k}|w{w}|sel"
            if key in R:
                continue
            U, info = train(M, F, A, L, k, mdas=(m == "mdas"), seed=0, iso_weight=w, **BUDGET)
            R[key] = dict(info=info, **ev(M, F, A, L, sel, ssrc, lambda sp, idx: iv_interchange(L, U, ssrc[sp][idx])))
            save()
            e = R[key]["entity"]
            print(f"{key}: E {e['score']:.2f} (C {e['cause']:.2f} I {e['iso']:.2f}) ctx {R[key]['context']['score']:.2f}", flush=True)
        best = max((R[f"{A}|L{L}|mdas|k{k}|w{w}|sel"]["entity"]["score"], k, w) for _, k, w in grid if _ == "mdas")
        _, kb, wb = best
        fresh = {sp: build_pool(F, sp, 200, seed={"entity": 2, "context": 3}[sp]) for sp in ("entity", "context")}
        fsrc = {sp: source_reps(M, p, [L])[L] for sp, p in fresh.items()}
        skey = f"{A}|L{L}|shortcuts_fresh"
        if skey not in R:
            tr_ents, tr = train_reps(M, F, [L])
            sae = SAE(L)
            with torch.no_grad():
                z = torch.cat([sae.enc(tr[L][i:i + 512].cuda()) for i in range(0, len(tr_ents), 512)])
            lat = probe_select(z, [F.labels[e][A] for e in tr_ents])[0]
            mu = mean_by_value(F, tr_ents, tr[L], A)
            zero = torch.zeros(M.d)
            vec = {sp: torch.stack([mu.get(F.labels[q.es][A], zero) - mu.get(F.labels[q.eb][A], zero) for q in p])
                   for sp, p in fresh.items()}
            R[skey] = {"full": ev(M, F, A, L, fresh, fsrc, lambda sp, idx: iv_set(L, fsrc[sp][idx])),
                       "meandiff": ev(M, F, A, L, fresh, fsrc, lambda sp, idx: iv_add(L, vec[sp][idx])),
                       "sae5": ev(M, F, A, L, fresh, fsrc, lambda sp, idx: iv_sae(L, sae, lat[5], fsrc[sp][idx])),
                       "sae50": ev(M, F, A, L, fresh, fsrc, lambda sp, idx: iv_sae(L, sae, lat[50], fsrc[sp][idx]))}
            save()
            print(f"{skey}: " + ", ".join(f"{m} {v['entity']['score']:.2f}" for m, v in R[skey].items()), flush=True)
        dk = max((128, 256), key=lambda k: R[f"{A}|L{L}|das|k{k}|w1.0|sel"]["entity"]["score"])
        for m, k, w, seeds in (("mdas", kb, wb, range(4)), ("das", dk, 1.0, range(2))):
            for seed in seeds:
                key = f"{A}|L{L}|{m}|k{k}|w{w}|fresh|s{seed}"
                if key in R:
                    continue
                U, info = train(M, F, A, L, k, mdas=(m == "mdas"), seed=seed, iso_weight=w, **BUDGET)
                R[key] = dict(info=info, **ev(M, F, A, L, fresh, fsrc, lambda sp, idx: iv_interchange(L, U, fsrc[sp][idx])))
                save()
                e = R[key]["entity"]
                print(f"{key}: E {e['score']:.2f} (C {e['cause']:.2f} I {e['iso']:.2f}) ctx {R[key]['context']['score']:.2f}", flush=True)
