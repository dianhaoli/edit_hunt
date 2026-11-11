"""Replay one recorded episode through the current tools and print every tool call whose error status differs from the
record, with the replayed error text, plus the re-graded reward.
  PYTHONPATH=. .venv/bin/python experiments/replay_debug.py results/suite/runs/<model>/<task>/<id>.json [...]"""
import glob
import json
import sys

from edithunt.env.grader import grade
from edithunt.env.instance import load_instance
from edithunt.env.tasks import make_env
from edithunt.model import Subject

S = Subject("Qwen/Qwen2.5-1.5B")
for p in sys.argv[1:]:
    r = json.load(open(p))
    inst = load_instance(glob.glob(f"results/suite/instances/*/{r['instance']}.private.json")[0])
    rec = [c["content"] for m in r["messages"] if m["role"] == "user" and isinstance(m["content"], list)
           for c in m["content"] if c.get("type") == "tool_result"]
    env = make_env(S, inst, blackbox=r["blackbox"])
    outs = [env.call(x["tool"], x["args"]) for x in r["tool_log"]]
    for i, (o, c) in enumerate(zip(outs[1:], rec), 1):
        if o.startswith('{"error"') != c.startswith('{"error"'):
            print(f"{r['instance']} call {i} {r['tool_log'][i]['tool']}: recorded {c[:160]} | replay {o[:300]}")
    g = grade(S, inst, env.submission)
    print(f"{r['instance']}: recorded reward {r['grade']['reward']:.4f}, replay {g['reward']:.4f}", flush=True)
