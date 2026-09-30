"""Scripted naive baselines on the suite instances (through the agent tools), for the task map.
  T1_easy, T3_consistency, T7_minimal: naive gradient (target-capital answer direction, KL-regularised)
  T2_keepstate: mean-diff and naive gradient
  T7_minimal: mean-diff at scale 1 (what a non-minimising agent submits)
  T6_handoff: prior guess round(0.8 * n_layers) (no tools)
Writes results/suite/baselines.json."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__) + "/..")
import json
from pathlib import Path

from edithunt.common import ROOT
from edithunt.env.baselines import agent_gradient, agent_meandiff
from edithunt.env.grader import grade
from edithunt.env.instance import load_instance
from edithunt.env.tasks import grade_report, make_env
from edithunt.model import Subject

RUNS = {"T1_easy": ["gradient"], "T2_keepstate": ["meandiff", "gradient"], "T3_consistency": ["gradient"],
        "T7_minimal": ["meandiff", "gradient"]}
AG = {"meandiff": agent_meandiff, "gradient": agent_gradient}
out = {}
S = Subject("Qwen/Qwen2.5-1.5B")
for task, ags in RUNS.items():
    for p in sorted((ROOT / "results/suite/instances" / task).glob("*.private.json")):
        inst = load_instance(p)
        for ag in ags:
            env = make_env(S, inst)
            try:
                AG[ag](env)
            except RuntimeError as e:
                print("error", e)
            g = grade(S, inst, env.submission)
            out.setdefault(f"{task}/{ag}", []).append({"instance": inst.id, "reward": g["reward"]})
            print(task, ag, inst.id, round(g["reward"], 3), flush=True)
for p in sorted((ROOT / "results/suite/instances/T6_handoff").glob("*.private.json")):
    inst = load_instance(p)
    n = {"Qwen/Qwen2.5-1.5B": 28, "Qwen/Qwen2.5-3B": 36, "google/gemma-2-2b": 26, "Qwen/Qwen2.5-7B": 28}[inst.model]
    g = grade_report(inst, {"report": {"layer": round(0.8 * n)}})
    out.setdefault("T6_handoff/prior_0.8depth", []).append({"instance": inst.id, "reward": g["reward"]})
(ROOT / "results/suite/baselines.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, "mean", round(sum(x["reward"] for x in v) / len(v), 3), "pass", sum(x["reward"] >= 0.5 for x in v), "/", len(v))
