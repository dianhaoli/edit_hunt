"""Design check of train_subspace recipes (no API): reference / DAS / modest-correct / exploit configs, graded."""
from __future__ import annotations

import json
import sys

from ..common import ROOT
from . import env as E
from .core import Model

OUT = ROOT / "results" / "screen" / "env" / "recipe_check.json"


def recipes(env, n_t, n_k, keep=True):
    i = env.inst
    ids = lambda a: [k for k, t in env.tid.items() if t.attr == a]
    r = [dict(n=n_t, templates=ids(i["target"]), answer=f"source:{i['target']}", differ=[i["target"]])]
    if keep:
        for a in i["keep"]:
            r.append(dict(n=n_k // len(i["keep"]), templates=ids(a), answer=f"base:{a}", differ=[i["target"]]))
    return r


CONFIGS = {"reference": dict(n_t=1500, n_k=1500, keep=True, k=256, steps=600),
           "das_k256": dict(n_t=3000, n_k=0, keep=False, k=256, steps=600),
           "das_k16": dict(n_t=3000, n_k=0, keep=False, k=16, steps=600),
           "modest_correct": dict(n_t=300, n_k=300, keep=True, k=64, steps=300),
           "das_k8": dict(n_t=3000, n_k=0, keep=False, k=8, steps=600),
           "das_k4": dict(n_t=3000, n_k=0, keep=False, k=4, steps=600),
           "das_k2": dict(n_t=3000, n_k=0, keep=False, k=2, steps=600),
           "das_k1": dict(n_t=3000, n_k=0, keep=False, k=1, steps=600)}

if __name__ == "__main__":
    M = Model()
    val = json.loads((ROOT / "results" / "screen" / "env" / "validate.json").read_text())
    E.KL_BUDGET = val["kl_budget"]
    R = json.loads(OUT.read_text()) if OUT.exists() else {}
    for iid in sys.argv[1:]:
        for name, c in CONFIGS.items():
            key = f"{iid}|{name}"
            if key in R:
                continue
            env = E.CityEnv(M, E.load_instance(iid))
            out = json.loads(env.call("train_subspace", {"layer": env.inst["layer"], "k": c["k"], "steps": c["steps"],
                                                         "recipes": recipes(env, c["n_t"], c["n_k"], c["keep"])}))
            if "error" in out:
                raise RuntimeError(out)
            env.call("submit", {"featurizer": out["featurizer"]})
            g = E.grade(M, env.inst, env.submission)
            R[key] = dict(train=out, grade=g)
            OUT.write_text(json.dumps(R, indent=1))
            print(f"{key}: score {g['score']:.2f} (C {g['cause']:.2f} I {g['iso']:.2f} KLf {g['kl_factor']:.2f}) | "
                  f"heldout loss {out['heldout_loss']} acc {out['heldout_answer_accuracy']} | {out['seconds']}s", flush=True)
