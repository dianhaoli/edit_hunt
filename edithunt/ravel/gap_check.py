"""Seed check for the gap-hunt winner (city_pop Language, L12, MDAS k128, 300 steps / 1200 pairs) and the missing
SAE shortcut at L12 (sae5/50, Stage-2 recipe). Fresh pools (entity seed 2, context seed 3); DAS k16/k128 and full /
mean-diff re-scored on the same fresh pools for a like-for-like gap."""
from __future__ import annotations

import json

import torch

from .core import Model, iv_add, iv_interchange, iv_set
from .das import train
from .evalkit import SCREEN, build_pool, load_survivor, mean_by_value, metrics, predict_pool, source_reps, train_reps
from .sae import SAE, iv_sae, probe_select

OUT = SCREEN / "gap_check.json"
A, L = "Language", 12

if __name__ == "__main__":
    M = Model()
    F = load_survivor("city_pop")
    R = json.loads(OUT.read_text()) if OUT.exists() else {"target": A, "layer": L}
    pools = {sp: build_pool(F, sp, 200, seed={"entity": 2, "context": 3}[sp]) for sp in ("entity", "context")}
    src = {sp: source_reps(M, p, [L])[L] for sp, p in pools.items()}
    ev = lambda mk: {sp: metrics(F, p, predict_pool(M, p, lambda idx: mk(sp, idx)), A) for sp, p in pools.items()}
    if "shortcuts" not in R:
        tr_ents, tr = train_reps(M, F, [L])
        sae = SAE(L)
        with torch.no_grad():
            z = torch.cat([sae.enc(tr[L][i:i + 512].cuda()) for i in range(0, len(tr_ents), 512)])
        lat, acc, _ = probe_select(z, [F.labels[e][A] for e in tr_ents])
        mu = mean_by_value(F, tr_ents, tr[L], A)
        zero = torch.zeros(M.d)
        vec = {sp: torch.stack([mu.get(F.labels[q.es][A], zero) - mu.get(F.labels[q.eb][A], zero) for q in p])
               for sp, p in pools.items()}
        R["shortcuts"] = {"full": ev(lambda sp, idx: iv_set(L, src[sp][idx])),
                          "meandiff": ev(lambda sp, idx: iv_add(L, vec[sp][idx])),
                          "sae5": ev(lambda sp, idx: iv_sae(L, sae, lat[5], src[sp][idx])),
                          "sae50": ev(lambda sp, idx: iv_sae(L, sae, lat[50], src[sp][idx]))}
        OUT.write_text(json.dumps(R, indent=1))
        print({m: round(v["entity"]["score"], 2) for m, v in R["shortcuts"].items()}, flush=True)
    R.setdefault("runs", {})
    for m, k in (("das", 16), ("das", 128), ("mdas", 128)):
        for seed in range(4 if m == "mdas" else 1):
            key = f"{m}|k{k}|s{seed}"
            if key in R["runs"]:
                continue
            U, info = train(M, F, A, L, k, mdas=(m == "mdas"), seed=seed, steps=300, n_train=1200, max_seconds=600)
            R["runs"][key] = dict(info=info, **ev(lambda sp, idx: iv_interchange(L, U, src[sp][idx])))
            OUT.write_text(json.dumps(R, indent=1))
            r = R["runs"][key]
            print(f"{key}: entity {r['entity']['score']:.2f} (C {r['entity']['cause']:.2f} I {r['entity']['iso']:.2f}) "
                  f"context {r['context']['score']:.2f}", flush=True)
