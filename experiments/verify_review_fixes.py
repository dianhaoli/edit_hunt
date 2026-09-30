"""Verification after the 2026-09-30 review fixes (normalised held-out guards, prefix-capture fix, nested city
matching, catch-all tool errors, T4 reference target): replay every Sonnet 5.5 run on Qwen2.5-1.5B through the
current tools and check (a) every tool call's error/no-error status matches the recorded transcript and (b) the
re-graded reward equals the recorded one. Runs on other subject models (T6 3B/gemma/7B) are listed as skipped."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__) + "/..")
import glob, json
from edithunt.env.grader import grade
from edithunt.env.instance import load_instance
from edithunt.env.tasks import make_env
from edithunt.model import Subject

S = Subject("Qwen/Qwen2.5-1.5B")
n = bad = skipped = 0
for p in sorted(glob.glob("results/suite/runs/claude-sonnet-5-5/*/*.json")):
    r = json.load(open(p))
    inst = load_instance(glob.glob(f"results/suite/instances/*/{r['instance']}.private.json")[0])
    if inst.model != S.name:
        skipped += 1; print("skip", r["instance"], inst.model); continue
    rec = [c["content"].startswith('{"error"') for m in r["messages"] if m["role"] == "user" and isinstance(m["content"], list)
           for c in m["content"] if c.get("type") == "tool_result"]
    env = make_env(S, inst, blackbox=r["blackbox"])
    outs = [env.call(x["tool"], x["args"]) for x in r["tool_log"]]
    rep = [o.startswith('{"error"') for o in outs[1:]]  # tool_log[0] is the describe_task in the first user message
    g = grade(S, inst, env.submission)
    ok = rep == rec and abs(g["reward"] - r["grade"]["reward"]) < 1e-9
    n += 1; bad += not ok
    print(("same " if ok else "DIFF ") + r["instance"] + (".bb" if r["blackbox"] else ""), r["grade"]["reward"], g["reward"],
          "" if rep == rec else f"errors rec {rec} vs replay {rep}", flush=True)
print(f"Sonnet replay: {n - bad}/{n} identical (tool error pattern + reward); {skipped} skipped (other subject models)")
