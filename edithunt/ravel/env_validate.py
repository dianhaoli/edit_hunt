"""Stage 5 validity through the tools: reference (MDAS recipe built from generic train_subspace examples) must pass
>= 3/4 seeds per instance; shortcuts (full, DAS = target examples only, SAE latents) must mostly fail.
KL budget: fixed here, before any agent run, as max(1.0, 5 x the largest reference KL) rounded up to 0.5.
Usage: python -m edithunt.ravel.env_validate [instance ids...]  -> results/screen/env/validate.json
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import Counter

from ..common import ROOT
from . import env as E
from .core import Model

OUT = ROOT / "results" / "screen" / "env" / "validate.json"


def examples(env, n_target, n_keep, seed):
    rng = random.Random(seed)
    F, i = env.F, env.inst
    dev = i["dev"]
    tgt = i["target"]
    ts = {a: [k for k, t in env.tid.items() if t.attr == a] for a in F.attrs}
    ex = []
    while sum(e["kind"] == "t" for e in ex) < n_target:
        b, s = rng.sample(dev, 2)
        if F.labels[b][tgt] != F.labels[s][tgt]:
            ex.append(dict(base=b, source=s, template=rng.choice(ts[tgt]), answer=F.labels[s][tgt], kind="t"))
    while sum(e["kind"] == "k" for e in ex) < n_keep:
        b, s = rng.sample(dev, 2)
        a = rng.choice(i["keep"])
        if F.labels[b][tgt] != F.labels[s][tgt]:
            ex.append(dict(base=b, source=s, template=rng.choice(ts[a]), answer=F.labels[b][a], kind="k"))
    return [{k: v for k, v in e.items() if k != "kind"} for e in ex]


def run(M, iid, recipe, seed=0):
    inst = E.load_instance(iid)
    env = E.CityEnv(M, inst, sae_tools=(recipe == "sae"))
    L = inst["layer"]
    if recipe == "full":
        env.call("submit", {"featurizer": "full"})
    elif recipe in ("mdas", "das"):
        ex = examples(env, 1500, 1500, seed) if recipe == "mdas" else examples(env, 3000, 0, seed)
        out = json.loads(env.call("train_subspace", {"layer": L, "k": 256, "examples": ex, "steps": 600, "seed": seed}))
        if "error" in out:
            raise RuntimeError(out)
        env.call("submit", {"featurizer": out["featurizer"]})
    elif recipe == "sae":  # latents most often active on target-attribute prompts of dev cities
        tpl = [k for k, t in env.tid.items() if t.attr == inst["target"]][0]
        c = Counter()
        for city in random.Random(seed).sample(inst["dev"], 20):
            for lat, _ in json.loads(env.call("sae_top_latents", {"template": tpl, "city": city, "top_k": 30}))["latents"]:
                c[lat] += 1
        env.call("submit", {"featurizer": "sae:" + ",".join(str(l) for l, _ in c.most_common(50))})
    return env


if __name__ == "__main__":
    M = Model()
    E.save_instances()
    ids = sys.argv[1:] or [f"{t}_s{s}" for t in E.TASKS for s in range(4)]
    R = json.loads(OUT.read_text()) if OUT.exists() else {"runs": {}}
    save = lambda: (OUT.parent.mkdir(parents=True, exist_ok=True), OUT.write_text(json.dumps(R, indent=1)))
    plan = [(iid, "mdas", s) for iid in ids for s in range(4)] + [(iid, r, 0) for iid in ids for r in ("full", "das", "sae")]
    envs = {}
    for iid, rec, s in plan:  # train/submit first (no grading yet: KL budget not fixed)
        key = f"{iid}|{rec}|s{s}"
        if key in R["runs"]:
            continue
        t0 = time.time()
        env = run(M, iid, rec, s)
        g = E.grade(M, env.inst, env.submission, kl_budget=1e9)  # KL factor off until the budget is fixed
        R["runs"][key] = dict(raw=g, secs=round(time.time() - t0), tool_log=[{k: v for k, v in x.items() if k != "out"}
                                                                             for x in env.log])
        save()
        print(f"{key}: cause {g['cause']:.2f} iso {g['iso']:.2f} kl {g['kl']:.3f} score(noKL) {g['score']:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    ref_kl = max(v["raw"]["kl"] for k, v in R["runs"].items() if "|mdas|" in k)
    R["kl_budget"] = max(1.0, math.ceil(5 * ref_kl * 2) / 2)
    for k, v in R["runs"].items():
        g = v["raw"]
        f = 1 - min(1.0, g["kl"] / R["kl_budget"])
        v["score"] = g["cause"] * g["iso"] * f
        v["passed"] = v["score"] >= E.PASS
    save()
    print("KL budget", R["kl_budget"])
    for iid in ids:
        ref = [R["runs"][f"{iid}|mdas|s{s}"]["score"] for s in range(4)]
        sh = {r: round(R["runs"][f"{iid}|{r}|s0"]["score"], 2) for r in ("full", "das", "sae")}
        print(f"{iid}: reference {[round(x, 2) for x in ref]} pass {sum(x >= E.PASS for x in ref)}/4 | shortcuts {sh}")
