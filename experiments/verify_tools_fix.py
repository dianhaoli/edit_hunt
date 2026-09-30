"""Verification after the 2026-09-30 tool fix (empty placeholders / clean model):
(1) calls that gpt-6.1-sol made with placeholder-filled optional fields now succeed;
(2) every Sonnet edit-task run replayed through the fixed tools reproduces its recorded reward exactly."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__) + "/..")
import glob, json
from edithunt.env.grader import grade
from edithunt.env.instance import load_instance
from edithunt.env.tasks import make_env
from edithunt.model import Subject

S = Subject("Qwen/Qwen2.5-1.5B")
inst = load_instance(sorted(glob.glob("results/suite/instances/T1_easy/*.private.json"))[0])
env = make_env(S, inst)
src, tgt = inst.dev_source, inst.dev_target
tests = [
    ("cache_mean", {"name": "s", "cities": src, "template": "fs1", "layer": 10, "layers": [], "position": "city_last", "model": "clean"}),
    ("cache_mean", {"name": "t", "cities": tgt, "template": "fs1", "layer": 10, "layers": [], "position": "city_last", "model": "clean"}),
    ("vec_op", {"name": "d", "expr": "t - s"}),
    ("eval_intervention", {"template": "fs1", "cities": src, "edits": [{"register": "d", "layer": 10, "position": "city_last",
                           "scale": 1.0, "basis": [], "center": ""}], "vector": "", "layer": 10, "position": "city_last",
                           "scale": 1.0, "n_generic": 2, "answers": "capitals"}),
    ("run_prompts", {"template": "fs1", "cities": src[:2], "prompts": [], "model": "clean", "answers": "capitals", "top_k": 3}),
]
ok = True
for name, args in tests:
    out = json.loads(env.call(name, args))
    print(name, "ERROR " + out["error"] if "error" in out else "ok", {k: out[k] for k in ("flip_rate",) if k in out})
    ok &= "error" not in out
print("placeholder calls:", "PASS" if ok else "FAIL")

bad = 0; n = 0
for p in sorted(glob.glob("results/suite/runs/claude-sonnet-5-5/*/*.json")):
    r = json.load(open(p))
    if r["suite_task"] not in ("T1_easy", "T2_keepstate", "T3_consistency", "T7_minimal"):
        continue
    inst = load_instance(glob.glob(f"results/suite/instances/*/{r['instance']}.private.json")[0])
    env = make_env(S, inst)
    for x in r["tool_log"]:
        env.call(x["tool"], x["args"])
    g = grade(S, inst, env.submission)
    same = abs(g["reward"] - r["grade"]["reward"]) < 1e-9
    n += 1; bad += not same
    print(("same " if same else "DIFF ") + r["instance"], r["grade"]["reward"], g["reward"], flush=True)
print(f"Sonnet replay: {n - bad}/{n} identical")
