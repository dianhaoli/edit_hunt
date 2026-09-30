"""Scripted baseline agents that use only the tool API, and a pass-rate harness.

  meandiff  (C1): mean(target dev cities) - mean(source dev cities) at city_last, pick the layer
                  (<= max_layer) with the best dev flip rate minus generic-KL penalty, submit.
  gradient  (C2): optimize_vector (Adam, first-token target log-prob + KL penalty) on dev cities.
  careful   (C2nk): optimize_vector at the top allowed layer on both dev templates with keep_cities = 4
                  cities of other states (must keep their own capital), norm capped at |mean-diff| of that
                  layer; on preserve_state tiers also keep_state_cities = dev source cities (Phase 3 hop2n).
                  The scripted version of "notice the leakage term and regularize against it" (Phase 4b).
  random        : random direction norm-matched to the mean-difference vector (control).

pass = reward >= threshold.

CLI: python -m edithunt.env.baselines --model Qwen/Qwen2.5-0.5B --tiers easy,medium --n 2
     [--instances_dir results/instances/Qwen2.5-0.5B] (loads *.private.json per tier if present)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..common import ROOT, save
import random

from ..data import CITIES, study_states
from ..metrics import rate
from ..model import Subject
from .grader import grade
from .instance import TIERS, generate, load_instance, save_instance
from .tools import ToolEnv


def _call(env: ToolEnv, tool: str, **kw) -> dict:
    out = json.loads(env.call(tool, kw))
    if "error" in out:
        raise RuntimeError(f"{tool}: {out['error']}")
    return out


def _setup(env: ToolEnv):
    d = _call(env, "describe_task")
    c = d["constraints"]
    top = c["max_layer"]
    layers = sorted({top, (2 * top) // 3, top // 3})
    tpl = next(iter(k for k in d["templates"] if k not in ("state_q", "country_q")))
    tgt = d["dev_target_cities"] or [x for x in CITIES[d["target"]]][:4]
    return d, c, layers, tpl, tgt


def meandiff_dirs(env: ToolEnv, layers: list[int], tpl: str, src: list[str], tgt: list[str]) -> None:
    for L in layers:
        _call(env, "cache_mean", name=f"s{L}", cities=src, template=tpl, layer=L)
        _call(env, "cache_mean", name=f"t{L}", cities=tgt, template=tpl, layer=L)
        _call(env, "vec_op", name=f"d{L}", expr=f"t{L} - s{L}")


def agent_meandiff(env: ToolEnv, **_) -> None:
    d, c, layers, tpl, tgt = _setup(env)
    src = d["dev_source_cities"]
    meandiff_dirs(env, layers, tpl, src, tgt)
    best = None
    for L in layers:
        for sc in (1.0, 1.5):
            r = _call(env, "eval_intervention", template=tpl, cities=src, vector=f"d{L}", layer=L,
                      position="city_last", scale=sc)
            score = r["flip_rate"] - r.get("kl_generic", 0) / c["kl_budget"]
            if best is None or score > best[0]:
                best = (score, L, sc)
    _, L, sc = best
    _call(env, "submit", edits=[{"register": f"d{L}", "layer": L, "position": "city_last", "scale": sc}])


def agent_gradient(env: ToolEnv, steps: int = 30, kl_weight: float = 1.0, **_) -> None:
    d, c, layers, tpl, tgt = _setup(env)
    L = layers[len(layers) // 2]
    tpls = [k for k in d["templates"] if k not in ("state_q", "country_q")]
    src = d["dev_source_cities"][: max(1, 16 // len(tpls))]
    _call(env, "optimize_vector", name="g", layer=L, position="city_last", dev_cities=src, templates=tpls,
          steps=steps, kl_weight=kl_weight)
    _call(env, "submit", edits=[{"register": "g", "layer": L, "position": "city_last", "scale": 1.0}])


def agent_careful(env: ToolEnv, steps: int = 50, seed: int = 0, **_) -> None:
    d, c, layers, tpl, tgt = _setup(env)
    L = c["max_layer"] if c["max_layer"] < 20 else 15  # Phase 4b: L8-15 best; beyond the band it degrades
    tpls = [k for k in d["templates"] if k not in ("state_q", "country_q")]
    src = d["dev_source_cities"][: max(1, 16 // len(tpls))]
    meandiff_dirs(env, [L], tpl, d["dev_source_cities"], tgt)
    nrm = _call(env, "vec_info", name=f"d{L}")["norm"]
    rng = random.Random(f"{seed}-{d['source']}-{d['target']}")
    others = [s for s in study_states(min_cities=5) if s not in (d["source"], d["target"])]
    keep = [x for st in rng.sample(others, 6) for x in CITIES[st][:3]]
    for _ in range(6):  # drop cities the tools refuse (grader-only cities), as an agent would
        if c.get("preserve_state"):  # v2 / hop-2 recipe (Phase 3 hop2n): source cities keep their state
            kw = dict(name="g", layer=L, position="city_last", dev_cities=src, templates=tpls, steps=steps,
                      keep_cities=keep[:4], max_norm=nrm, keep_state_cities=d["dev_source_cities"])
        else:  # v1 recipe: best on medium in the first calibration (lower country/generic KL than v2)
            kw = dict(name="g", layer=L, position="city_last", dev_cities=src, templates=tpls[:1], steps=30,
                      keep_cities=keep[:8], max_norm=nrm)
        o = json.loads(env.call("optimize_vector", kw))
        if "error" not in o:
            break
        bad = [x for x in keep if f"'{x}'" in o["error"]]
        if not bad:
            raise RuntimeError(f"optimize_vector: {o['error']}")
        keep = [x for x in keep if x not in bad]
    else:
        raise RuntimeError("could not find accessible keep cities")
    _call(env, "submit", edits=[{"register": "g", "layer": L, "position": "city_last", "scale": 1.0}])


def agent_random(env: ToolEnv, seed: int = 0, **_) -> None:
    d, c, layers, tpl, tgt = _setup(env)
    L = layers[-1]
    meandiff_dirs(env, [L], tpl, d["dev_source_cities"], tgt)
    _call(env, "vec_op", name="r", expr=f"randn({seed}) * norm(d{L})")
    _call(env, "submit", edits=[{"register": "r", "layer": L, "position": "city_last", "scale": 1.0}])


AGENTS = {"meandiff": agent_meandiff, "gradient": agent_gradient, "careful": agent_careful, "random": agent_random}


def run_agent(S: Subject, inst, agent: str, bs: int = 16, **kw) -> dict:
    env = ToolEnv(S, inst, bs)
    t0 = time.time()
    try:
        AGENTS[agent](env, **kw)
        err = None
    except RuntimeError as e:
        err = str(e)
    g = grade(S, inst, env.submission, bs) if env.done else {"valid": False, "reason": err or "no submission", "reward": 0.0}
    return {"instance": inst.id, "agent": agent, "error": err, "budget_used": env.used, "reward": g["reward"],
            "grade": g, "secs": round(time.time() - t0, 1)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--tiers", default="easy,medium,hard")
    ap.add_argument("--agents", default="meandiff,gradient,careful,random")
    ap.add_argument("--n", type=int, default=3, help="instances per tier")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--handoff", type=int, default=None)
    ap.add_argument("--instances_dir", default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_cities", type=int, default=None)
    ap.add_argument("--pairs", default="", help="e.g. 'Texas:California,Ohio:Oregon' (else random)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default="phase6_baselines")
    a = ap.parse_args()
    S = Subject(a.model, device=a.device)
    h = a.handoff if a.handoff is not None else S.n_layers // 2
    rows, summary = [], {}
    for tier in a.tiers.split(","):
        assert tier in TIERS, tier
        d = Path(a.instances_dir) / tier if a.instances_dir else None
        if d and d.exists():
            insts = [load_instance(p) for p in sorted(d.glob("*.private.json"))][: a.n]
        else:
            pairs = [tuple(p.split(":")) for p in a.pairs.split(",") if p] or None
            insts = generate(S, tier, a.n, a.seed, h, pairs=pairs, bs=a.bs, max_cities=a.max_cities)
            for i in insts:
                save_instance(i, ROOT / "results" / "instances" / a.model.split("/")[-1] / tier)
        for ag in a.agents.split(","):
            rs = [run_agent(S, i, ag, a.bs, steps=a.steps, seed=a.seed) for i in insts]
            for r in rs:
                g = r["grade"]
                print(f"{tier:6s} {ag:9s} {r['instance']:40s} reward {r['reward']:.3f} F {g.get('F', float('nan')):.2f} "
                      f"KL {g.get('kl_mean', float('nan')):.3f} used {r['budget_used']} {r['error'] or ''}", flush=True)
            rows += [r | {"tier": tier} for r in rs]
            rw = [r["reward"] for r in rs]
            summary[f"{tier}/{ag}"] = {"pass": rate([x >= a.threshold for x in rw]),
                                       "mean_reward": sum(rw) / len(rw) if rw else float("nan")}
            print(f"== {tier}/{ag}: pass {summary[f'{tier}/{ag}']['pass']}", flush=True)
    p = save(a.model, a.out, vars(a), {"summary": summary, "rows": rows})
    print("saved", p)


if __name__ == "__main__":
    main()
