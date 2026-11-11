"""T2 v2.1 on the 10 pilot instances: references (3 seeds) + scripted-baseline library + H3 budget curve, all
through the agent tools, graded with grader v2.1 (hard + soft). Submissions saved for re-grading.
Library (fixed scripts; public info + accessible cities only; first five from the AUDIT_T2V2 scripted_gate.py):
  md_mid_x1        gate (40-60% ramp) + mean-diff push, no optimisation
  opt_target100    gate + push trained on the target only
  opt_keep{S}      gate + push trained with keep examples (state_q on dev, own capital on 6 other cities), S steps
  opt_keep100_s{1,2}  same, other seeds (choice of other-state cities)
  opt_keep{10,20}_lr02  short runs with a 4x learning rate (budget curve)
  opt_keep100_more self-named extra source cities (up to 8 accessible, from the extended list) + 8 other cities
  ungated_keep100  no gate: plain additive vector with the same keep examples (T2 v1 recipe)
  ref_s{0,1,2}     tasks.ref_ravel_gated (150 steps, ramp 30-60%)
Rows -> results/suite/t2v21/library.jsonl; submissions -> results/suite/t2v21/subs/<instance>__<method>.json
  PYTHONPATH=. .venv/bin/python experiments/t2v21_library.py [--methods a,b] [--shard 0/1]"""
import argparse
import glob
import json
import random
import time

from edithunt.common import ROOT
from edithunt.data import CAPITALS, CITIES, CITIES_EXT, CITY2STATE_EXT, study_states
from edithunt.model import Subject
from edithunt.env import ravel as R
from edithunt.env.instance import load_instance, save_instance
from edithunt.env.tasks import _accessible, _call, make_env, ref_ravel_gated

OUT = ROOT / "results" / "suite" / "t2v21"
IDIR = ROOT / "results" / "suite" / "instances" / "T2_keepstate_v21"


def setup(env, seed=0, n_oth=8, extra_src=0):
    d = _call(env, "describe_task")
    L = d["constraints"]["layer"]
    tpls = [k for k in d["templates"] if k not in ("state_q", "country_q")]
    src = list(d["dev_source_cities"])
    if extra_src:  # self-named source cities (any accessible city of the source state the agent can name)
        more = [c for c in _accessible(env, CITIES_EXT[d["source"]]) if c not in src]
        src += random.Random(f"more-{seed}-{d['source']}").sample(more, min(extra_src, len(more)))
    rng = random.Random(f"audit-{seed}-{d['source']}-{d['target']}")
    others = [s for s in study_states(min_cities=5) if s not in (d["source"], d["target"])]
    oth = [x for st in rng.sample(others, 12) for x in _accessible(env, CITIES[st])[:1]][:n_oth]
    for t in tpls:
        for nm, cs in (("s", src), ("o", oth), ("t", d["dev_target_cities"])):
            _call(env, "cache_mean", name=f"{nm}_{t}", cities=cs, template=t, layer=L)
    _call(env, "vec_op", name="k", expr="normalize(" + " + ".join(f"s_{t}" for t in tpls) + " - (" +
          " + ".join(f"o_{t}" for t in tpls) + "))")
    _call(env, "vec_op", name="md", expr="(" + " + ".join(f"t_{t} - s_{t}" for t in tpls) + f") / {len(tpls)}")
    ms = sum(_call(env, "project", name="k", cities=src[:16], template=t, layer=L)["mean"] for t in tpls) / len(tpls)
    mo = sum(_call(env, "project", name="k", cities=oth, template=t, layer=L)["mean"] for t in tpls) / len(tpls)
    return d, L, tpls, src, oth, ms, mo


def gate(ms, mo, a=0.4, b=0.6):
    return {"key": "k", "lo": mo + a * (ms - mo), "hi": mo + b * (ms - mo)}


def submit(env, L, reg, g=None, scale=1.0):
    e = {"register": reg, "layer": L, "position": "city_last", "scale": scale}
    if g:
        e["scale_from"] = g
    _call(env, "submit", edits=[e])


def m_md_mid(env):
    d, L, tpls, src, oth, ms, mo = setup(env)
    submit(env, L, "md", gate(ms, mo))


def m_opt_target(env, steps=100):
    d, L, tpls, src, oth, ms, mo = setup(env)
    g = gate(ms, mo)
    _call(env, "optimize_vector", name="v", layer=L, position="city_last", dev_cities=src, templates=tpls, steps=steps,
          scale_from=g)
    submit(env, L, "v", g)


def keep_examples(d, src, oth, tpls, n_src=4, n_oth=6):
    ex = [{"template": "state_q", "city": c, "answer": d["source"]} for c in src[:n_src]]
    ex += [{"template": t, "city": c, "answer": CAPITALS[CITY2STATE_EXT[c]]} for c in oth[:n_oth] for t in tpls]
    return ex[:16]


def m_opt_keep(env, steps=100, seed=0, lr=0.05, gated=True, extra_src=0, n_oth=8):
    d, L, tpls, src, oth, ms, mo = setup(env, seed, n_oth=n_oth, extra_src=extra_src)
    g = gate(ms, mo) if gated else None
    ex = keep_examples(d, src, oth, tpls, n_src=4 if not extra_src else 8, n_oth=6 if not extra_src else 4)
    kw = dict(scale_from=g) if g else {}
    _call(env, "optimize_vector", name="v", layer=L, position="city_last", dev_cities=src[:8], templates=tpls,
          steps=steps, lr=lr, extra_examples=ex, **kw)
    submit(env, L, "v", g)


METHODS = {"md_mid_x1": m_md_mid, "opt_target100": m_opt_target,
           **{f"opt_keep{s}": (lambda s: lambda e: m_opt_keep(e, s))(s) for s in (5, 10, 20, 40, 100, 150)},
           "opt_keep10_lr02": lambda e: m_opt_keep(e, 10, lr=0.2), "opt_keep20_lr02": lambda e: m_opt_keep(e, 20, lr=0.2),
           "opt_keep100_s1": lambda e: m_opt_keep(e, 100, 1), "opt_keep100_s2": lambda e: m_opt_keep(e, 100, 2),
           "opt_keep100_more": lambda e: m_opt_keep(e, 100, extra_src=4),
           "ungated_keep100": lambda e: m_opt_keep(e, 100, gated=False),
           **{f"ref_s{s}": (lambda s: lambda e: ref_ravel_gated(e, seed=s))(s) for s in (0, 1, 2)}}


def pilot_v21() -> list:
    """The 10 v2 pilot instances as grader-v2.1 copies (written once to IDIR)."""
    IDIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(glob.glob(str(ROOT / "results/suite/instances/T2_keepstate_v2/*.private.json"))):
        q = IDIR / (load_instance(p).id + "-v21.private.json")
        if not q.exists():
            x = R.to_v21(load_instance(p))
            x.extra["reference_v2"] = x.extra.pop("reference", None)
            save_instance(x, IDIR)
        out.append(load_instance(q))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    K, N = map(int, a.shard.split("/"))
    ms = [m for m in a.methods.split(",") if m] or list(METHODS)
    (OUT / "subs").mkdir(parents=True, exist_ok=True)
    out = OUT / f"library.{K}.jsonl"
    done = {(r["id"], r["method"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    f = out.open("a")
    S = Subject("Qwen/Qwen2.5-1.5B")
    jobs = [(inst, m) for m in ms for inst in pilot_v21()]
    for i, (inst, m) in enumerate(jobs):
        if i % N != K or (inst.id, m) in done:
            continue
        t0 = time.time()
        env = make_env(S, inst)
        err = None
        try:
            METHODS[m](env)
        except RuntimeError as e:
            err = str(e)
        g = R.grade_ravel(S, inst, env.submission) if env.submission else {"valid": False}
        (OUT / "subs" / f"{inst.id}__{m}.json").write_text(json.dumps(env.submission))
        r = {"id": inst.id, "method": m, "err": err, "fwd": env.used, "secs": round(time.time() - t0, 1),
             "R": g.get("reward", 0.0), "R_soft": g.get("soft", {}).get("reward", 0.0)}
        if g.get("valid"):
            r |= {k: g[k]["rate"] for k in ("Cause", "Iso_state", "Iso_other")} | {"kl": g["kl_mean"], "soft": g["soft"]}
        f.write(json.dumps(r) + "\n"); f.flush()
        print(inst.id, m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                           if k in ("R", "R_soft", "Cause", "Iso_state", "Iso_other", "kl", "fwd", "err")}, flush=True)


if __name__ == "__main__":
    main()
