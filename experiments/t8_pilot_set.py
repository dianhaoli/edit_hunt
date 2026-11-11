"""T8_undo agent-pilot set (amendment A5, LAB_NOTEBOOK 2026-10-02). The strict generator filter keeps none of the
Minnesota->Tennessee candidates (the 4-script library solves single plants), so pilot instances are chosen by script
difficulty instead (label: NOT filter-kept): hop2 L8 (detect v1/v3 solve), spread L8 (only v3 solves), two L8 (one-shot
detectors fix one plant), state L16 (v1 solves). Each is rebuilt, saved to results/suite/instances/T8_undo/, and
re-scored on the saved file: solver (inverse_post), patch_L20, detect v1/v2/v3 -> results/suite/t8_undo/pilot.jsonl.
  PYTHONPATH=. .venv/bin/python experiments/t8_pilot_set.py"""
import json

from edithunt.common import ROOT
from edithunt.env import undo as U
from edithunt.env.instance import load_instance, save_instance
from edithunt.env.tasks import make_env
from edithunt.model import Subject

PILOT = [("Minnesota", "Tennessee", "hop2", 8), ("Minnesota", "Tennessee", "spread", 8),
         ("Minnesota", "Tennessee", "two", 8), ("Minnesota", "Tennessee", "state", 16)]
D = ROOT / "results" / "suite" / "instances" / "T8_undo"
OUT = ROOT / "results" / "suite" / "t8_undo" / "pilot.jsonl"
SCRIPTS = {"patch_L20": U.script_patch, "detect_v1": U.script_detect,
           "detect_v2": lambda env: U.script_detect(env, ratio=True), "detect_v3": U.script_detect3}


def main():
    S = Subject("Qwen/Qwen2.5-1.5B")
    done = {(r["id"], r["method"]) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
    f = OUT.open("a")
    for src, tgt, pt, L in PILOT:
        iid = f"T8_undo-{src}-{tgt}-{pt}-L{L}-s0".replace(" ", "_")
        p = D / f"{iid}.private.json"
        if not p.exists():
            inst = U.make_instance(S, src, tgt, L, pt)
            inst.extra["selection"] = "agent pilot by script difficulty (not generator-kept)"
            save_instance(inst, D)
        inst = load_instance(p)
        if (inst.id, "solver") not in done:
            g = U.grade_undo(S, inst, U.inverse_post(S, inst))
            f.write(json.dumps({"id": inst.id, "method": "solver", "R": g["reward"], "R_src": g["restoration_src"],
                                "boot_sd": g["boot_sd"]}) + "\n"); f.flush()
        for m, fn in SCRIPTS.items():
            if (inst.id, m) in done:
                continue
            env = make_env(S, inst)
            info, err = {}, None
            try:
                info = fn(env) or {}
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:300]
            g = U.grade_undo(S, inst, env.submission) if env.submission else {"reward": 0.0}
            r = {"id": inst.id, "method": m, "R": g["reward"], "R_src": g.get("restoration_src"), "fwd": env.used,
                 "err": err, **{k: v for k, v in info.items() if k != "anomaly"}}
            f.write(json.dumps(r) + "\n"); f.flush()
            print(inst.id, m, round(r["R"], 3), err or "", flush=True)


if __name__ == "__main__":
    main()
