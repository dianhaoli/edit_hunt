"""T8 v0.2 instance set (LAB_NOTEBOOK 2026-10-02): env fixes from the agent audit (turns shown, private cities skipped
per call, opaque ids). 4 MN->TN pilot instances (copied from v0.1) + 4 NC->KY (hop2 L8, spread L8, two L8, state L16).
Each saved file is scored: solver (inverse_post) + patch_L20 + detect v1/v2/v3 -> results/suite/t8_undo/pilot_v02.jsonl.
  PYTHONPATH=. .venv/bin/python experiments/t8_v02_set.py"""
import copy
import hashlib
import json

from edithunt.common import ROOT
from edithunt.env import undo as U
from edithunt.env.instance import load_instance, save_instance
from edithunt.env.tasks import make_env
from edithunt.model import Subject

D1 = ROOT / "results/suite/instances/T8_undo"
D2 = ROOT / "results/suite/instances/T8_undo_v02"
OUT = ROOT / "results/suite/t8_undo/pilot_v02.jsonl"
NEW = [("North Carolina", "Kentucky", "hop2", 8), ("North Carolina", "Kentucky", "spread", 8),
       ("North Carolina", "Kentucky", "two", 8), ("North Carolina", "Kentucky", "state", 16)]
SCRIPTS = {"patch_L20": U.script_patch, "detect_v1": U.script_detect,
           "detect_v2": lambda env: U.script_detect(env, ratio=True), "detect_v3": U.script_detect3}


def v02(inst):
    x = copy.deepcopy(inst)
    x.extra["name"] = inst.id
    x.id = "T8u-" + hashlib.sha1(inst.id.encode()).hexdigest()[:8]
    x.constraints = dict(inst.constraints, show_turns=True, skip_private=True)
    x.extra["version"] = "v0.2"
    return x


def main():
    S = Subject("Qwen/Qwen2.5-1.5B")
    D2.mkdir(parents=True, exist_ok=True)
    insts = [load_instance(p) for p in sorted(D1.glob("*.private.json"))]
    for src, tgt, pt, L in NEW:
        name = f"T8_undo-{src}-{tgt}-{pt}-L{L}-s0".replace(" ", "_")
        if not any(p.name.startswith("T8u-") and json.loads(p.read_text())["extra"].get("name") == name
                   for p in D2.glob("*.private.json")):
            insts.append(U.make_instance(S, src, tgt, L, pt))
        else:
            insts.append(None)
    done = {(r["id"], r["method"]) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
    f = OUT.open("a")
    for inst in insts:
        if inst is None:
            continue
        x = v02(inst) if not inst.id.startswith("T8u-") else inst
        if not (D2 / f"{x.id}.private.json").exists():
            save_instance(x, D2)
        x = load_instance(D2 / f"{x.id}.private.json")
        if (x.id, "solver") not in done:
            g = U.grade_undo(S, x, U.inverse_post(S, x))
            f.write(json.dumps({"id": x.id, "name": x.extra["name"], "method": "solver", "R": g["reward"],
                                "R_src": g["restoration_src"], "boot_sd": g["boot_sd"]}) + "\n"); f.flush()
            print(x.id, x.extra["name"], "solver", round(g["reward"], 3), flush=True)
        for m, fn in SCRIPTS.items():
            if (x.id, m) in done:
                continue
            env = make_env(S, x)
            info, err = {}, None
            try:
                info = fn(env) or {}
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:300]
            g = U.grade_undo(S, x, env.submission) if env.submission else {"reward": 0.0}
            f.write(json.dumps({"id": x.id, "name": x.extra["name"], "method": m, "R": g["reward"], "err": err,
                                "fwd": env.used, **{k: v for k, v in info.items() if k != "anomaly"}}) + "\n"); f.flush()
            print(x.id, x.extra["name"], m, round(g["reward"], 3), err or "", flush=True)


if __name__ == "__main__":
    main()
