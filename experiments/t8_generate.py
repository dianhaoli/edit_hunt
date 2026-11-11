"""T8_undo instance generation with the built-in script filter (edithunt/env/generator.py; bars pre-registered in
LAB_NOTEBOOK 2026-10-02). Candidates = source/target pairs x plant type x plant layer. Solver = privileged inverse_post
(true -v, gate fitted on post-plant activations; deterministic, one seed). Library (through the agent tools):
patch_L20 (T2 recipe aimed at visible truths), detect_v1 (sweep -> residual-update anomaly vs control states ->
gated inverse at every position), detect_v2 (same with a scale-free ratio score; added after the probe). Kept instances -> results/suite/instances/T8_undo/; report + cache ->
results/suite/t8_undo/.
  PYTHONPATH=. .venv/bin/python experiments/t8_generate.py [--pairs N] [--layers 6,10,14,18] [--types state,hop2]"""
import argparse
import json

from edithunt.common import ROOT
from edithunt.env import undo as U
from edithunt.env.generator import Family, screen
from edithunt.env.instance import save_instance
from edithunt.env.tasks import make_env
from edithunt.model import Subject

OUT = ROOT / "results" / "suite" / "t8_undo"
PAIRS = [("Minnesota", "Tennessee"), ("North Carolina", "Kentucky"), ("Wisconsin", "Mississippi"),
         ("Pennsylvania", "California"), ("Texas", "Ohio"), ("Virginia", "Oregon"), ("Indiana", "Colorado"),
         ("Oklahoma", "New Jersey")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--layers", default="8,16")
    ap.add_argument("--types", default="state,hop2,spread,two")
    a = ap.parse_args()
    S = Subject("Qwen/Qwen2.5-1.5B")
    insts = {}

    def candidates():
        for src, tgt in PAIRS[:a.pairs]:
            for pt in a.types.split(","):
                for L in map(int, a.layers.split(",")):
                    inst = U.make_instance(S, src, tgt, L, pt)
                    insts[inst.id] = inst
                    yield inst

    def solve(inst, seed):
        g = U.grade_undo(S, inst, U.inverse_post(S, inst))
        return {"R": g["reward"], "R_src": g["restoration_src"], "boot_sd": g["boot_sd"], "by_context": g["by_context"]}

    def tool_script(fn):
        def run(inst):
            env = make_env(S, inst)
            info, err = {}, None
            try:
                info = fn(env) or {}
            except Exception as e:  # a failing script scores what it submitted (usually nothing); the error is kept
                err = f"{type(e).__name__}: {e}"[:300]
            g = U.grade_undo(S, inst, env.submission) if env.submission else {"reward": 0.0}
            return {"R": g["reward"], "R_src": g.get("restoration_src"), "fwd": env.used, "err": err, **info,
                    "submission": env.submission}
        return run

    fam = Family("T8_undo", candidates, solve,
                 {"patch_L20_sw2": tool_script(U.script_patch), "detect_v1_sw2": tool_script(U.script_detect),
                  "detect_v2_sw2": tool_script(lambda env: U.script_detect(env, ratio=True)),
                  "detect_v3": tool_script(U.script_detect3)},
                 noise=lambda inst, r: r.get("boot_sd"), seeds=(0,))
    res = screen(fam, OUT)
    d = ROOT / "results" / "suite" / "instances" / "T8_undo"
    for inst in res["kept"]:
        save_instance(inst, d)
    print(json.dumps({k: res[k] for k in ("n_candidates", "n_kept")}))


if __name__ == "__main__":
    main()
