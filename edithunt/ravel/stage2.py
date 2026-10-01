"""Stage 2: cheap methods on every surviving family / target attribute.

Layers: full-vector Cause sweep over all layers (Entity split, 48 items per prompt attribute), then for each target
depth in (0.25, 0.45, 0.65) the layer with the highest mean full-vector Cause within +-2 layers of round(depth*n).
Methods per layer (both splits, 200 items per prompt attribute):
  full       h' = h_source (target-agnostic)
  random16   interchange in a random 16-dim orthonormal subspace (target-agnostic; seed = layer)
  meandiff   h' = h_b + mu_A[val(e_s)] - mu_A[val(e_b)], means over train entities x train templates
             (categorical attributes only; not verbs)
  sae5/50    Gemma Scope latent interchange, latents = top-|w| of an L1 probe for A on train reps; base's
             reconstruction error kept
Output: results/screen/stage2/{family}.json (checkpointed after every layer).
Usage: python -m edithunt.ravel.stage2 [family ...]
"""
from __future__ import annotations

import json
import sys
import time

import torch

from .core import Model, iv_add, iv_interchange, iv_set
from .evalkit import (SCREEN, all_metrics, build_pool, load_survivor, mean_by_value, metrics, predict_pool,
                      source_reps, train_reps)
from .sae import SAE, iv_sae, probe_select, verb_class

OUT = SCREEN / "stage2"
DEPTHS = (0.25, 0.45, 0.65)
N_SWEEP, N_EVAL = 48, 200


def pick_layers(sweep: dict, n_layers: int) -> list[int]:
    out = []
    for d in DEPTHS:
        c = round(d * n_layers)
        cand = [L for L in range(max(0, c - 2), min(n_layers - 1, c + 2) + 1) if L not in out]
        out.append(max(cand, key=lambda L: (sweep[str(L)], -abs(L - c))))
    return out


def layer_sweep(M, F):
    pool = build_pool(F, "entity", N_SWEEP, seed=99)
    layers = list(range(M.n_layers))
    src = source_reps(M, pool, layers)
    res = {}
    for L in layers:
        preds = predict_pool(M, pool, lambda idx, L=L: iv_set(L, src[L][idx]))
        m = all_metrics(F, pool, preds)
        res[str(L)] = sum(v["cause"] for v in m.values()) / len(m)
    return res


def run_family(M: Model, name: str):
    t0 = time.time()
    F = load_survivor(name)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    R = json.loads(path.read_text()) if path.exists() else {"family": name, "targets": F.targets, "attrs": F.attrs}
    if "sweep" not in R:
        R["sweep"] = layer_sweep(M, F)
        R["layers"] = pick_layers(R["sweep"], M.n_layers)
        path.write_text(json.dumps(R, indent=1))
        print(f"  {name} sweep {[round(R['sweep'][str(L)], 2) for L in range(M.n_layers)]} -> {R['layers']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    layers = R["layers"]
    R.setdefault("results", {})
    R.setdefault("sae", {})
    pools = {sp: build_pool(F, sp, N_EVAL, seed={"entity": 0, "context": 1}[sp]) for sp in ("entity", "context")}
    R["pool_sizes"] = {sp: len(p) for sp, p in pools.items()}
    tr_ents, tr = train_reps(M, F, layers)
    for L in layers:
        if all(str(L) in R["results"].get(sp, {}) for sp in pools):
            continue
        sae = SAE(L)
        lat = {}
        for a in F.targets:
            if name == "verb":
                y = [verb_class(e, F.labels[e][a], a) for e in tr_ents]
            else:
                y = [F.labels[e][a] for e in tr_ents]
            with torch.no_grad():
                z = torch.cat([sae.enc(tr[L][i:i + 512].cuda()) for i in range(0, len(tr_ents), 512)])
            sel, acc, nnz = probe_select(z, y)
            lat[a] = sel
            R["sae"][f"{a}@L{L}"] = dict(sae_id=sae.id, probe_train_acc=acc, nonzero=nnz, top50=sel[50])
            del z
        means = {a: mean_by_value(F, tr_ents, tr[L], a) for a in F.targets} if name != "verb" else {}
        g = torch.Generator().manual_seed(L)
        Urand = torch.linalg.qr(torch.randn(M.d, 16, generator=g))[0]
        for sp, pool in pools.items():
            src = source_reps(M, pool, [L])[L]
            res = {}
            p = predict_pool(M, pool, lambda idx: iv_set(L, src[idx]))
            res["full"] = all_metrics(F, pool, p)
            p = predict_pool(M, pool, lambda idx: iv_interchange(L, Urand, src[idx]))
            res["random16"] = all_metrics(F, pool, p)
            for a in F.targets:
                if a in means:
                    mu = means[a]
                    zero = torch.zeros(M.d)
                    vec = torch.stack([mu.get(F.labels[q.es][a], zero) - mu.get(F.labels[q.eb][a], zero)
                                       for q in pool])
                    p = predict_pool(M, pool, lambda idx: iv_add(L, vec[idx]))
                    res.setdefault("meandiff", {})[a] = metrics(F, pool, p, a)
                for k in (5, 50):
                    p = predict_pool(M, pool, lambda idx: iv_sae(L, sae, lat[a][k], src[idx]))
                    res.setdefault(f"sae{k}", {})[a] = metrics(F, pool, p, a)
            R["results"].setdefault(sp, {})[str(L)] = res
            path.write_text(json.dumps(R, indent=1))
            best = {a: max((res[m][a]["score"], m) for m in res if a in res[m]) for a in F.targets}
            print(f"  {name} L{L} {sp}: " + "; ".join(f"{a}: best {m} {s:.2f}" for a, (s, m) in best.items()) +
                  f" ({time.time() - t0:.0f}s)", flush=True)
        del sae
        torch.cuda.empty_cache()
    return R


def main(names):
    M = Model()
    for n in names:
        run_family(M, n)


if __name__ == "__main__":
    main(sys.argv[1:])
