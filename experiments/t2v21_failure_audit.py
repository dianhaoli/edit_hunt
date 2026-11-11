"""T2 v2.1 agent failures: skill or environment? (rule fixed in LAB_NOTEBOOK before running). Per episode: re-grade the
recorded submission with per-item detail (grader consistency), bootstrap SD of R by city (200 draws), components.
Classes: ENV_GRADER (|regrade - recorded| > 0.02); PASS; NOISE (|R - bar| < 2 x bootstrap SD); LOW_CEILING (v2 ceiling
< 1.25 x bar, i.e. the instance barely admits the bar); else SKILL:<component> where component = the smallest of
Cause / Iso_state / Iso_other (Cause low = gate too strict or push too weak; Iso_state low = the state moved;
Iso_other low = gate too loose). Tool errors and end causes are reported alongside.
  PYTHONPATH=. .venv/bin/python experiments/t2v21_failure_audit.py"""
import glob
import json
import random
import statistics as st

from edithunt.common import ROOT
from edithunt.env import ravel as R
from edithunt.env.instance import load_instance
from edithunt.model import Subject

S = Subject("Qwen/Qwen2.5-1.5B")
lib = json.loads((ROOT / "results/suite/t2v21/library_summary.json").read_text())["instances"]
out = (ROOT / "results/suite/t2v21/failure_audit.jsonl").open("a")


def boot(items, n=200, seed=0):
    from experiments.t2v21_soft_analysis import parts
    rng = random.Random(seed)
    cs = sorted({x[0] for x in items["cause"]}); os_ = sorted({x[0] for x in items["iso_other"]})
    return st.pstdev(parts(items, cities=[rng.choice(cs) for _ in cs], ocities=[rng.choice(os_) for _ in os_])[0]
                     for _ in range(n))


for p in sorted(glob.glob(str(ROOT / "results/suite/runs/*/T2_keepstate_v21/*.json"))):
    rec = json.loads(open(p).read())
    inst = load_instance(ROOT / f"results/suite/instances/T2_keepstate_v21/{rec['instance']}.private.json")
    Rrec = rec["grade"].get("reward", 0.0)
    bar = 0.8 * lib[inst.id]["ref_mean"]
    ceil = lib[inst.id]["ceiling_v2"]
    outs = [c.get("output") or c.get("content") or "" for m in rec["messages"] if m["role"] == "user"
            and isinstance(m["content"], list) for c in m["content"]]
    errs = {"refusal": sum('"not accessible"' in o for o in outs), "internal": sum("internal error" in o for o in outs),
            "other": sum(o.startswith('{"error"') and "not accessible" not in o and "internal error" not in o for o in outs)}
    row = {"record": p.split("/")[-1], "model": rec["agent_model"], "R": Rrec, "bar": bar, "ceiling_v2": ceil,
           "end": rec["end"], "errors": errs}
    if rec.get("submission_full"):
        g = R.grade_ravel(S, inst, rec["submission_full"], detail=True)
        sd = boot(g["items"])
        comp = {k: g[k]["rate"] for k in ("Cause", "Iso_state", "Iso_other")}
        row |= {"regraded": g["reward"], "boot_sd": sd, **comp, "kl": g["kl_mean"],
                "gate": [{"lo": e.get("lo"), "hi": e.get("hi"), "norm": e.get("norm")} for e in g.get("edits", [])]}
        if abs(g["reward"] - Rrec) > 0.02:
            cls = "ENV_GRADER"
        elif Rrec >= bar:
            cls = "PASS"
        elif abs(Rrec - bar) < 2 * sd:
            cls = "NOISE"
        elif ceil is not None and ceil < 1.25 * bar:
            cls = "LOW_CEILING"
        else:
            cls = "SKILL:" + min(comp, key=comp.get)
    else:
        cls = "NO_SUBMISSION:" + str(rec["end"])
    row["class"] = cls
    out.write(json.dumps(row) + "\n"); out.flush()
    print(row["record"][13:], cls, "R %.3f bar %.3f" % (Rrec, bar), {k: round(row[k], 3) for k in ("regraded", "boot_sd", "Cause", "Iso_state", "Iso_other") if k in row}, errs, flush=True)
