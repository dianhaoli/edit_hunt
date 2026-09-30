"""Shortcut baselines for construct validity (no API). Run from repo root."""
import sys, os, json, glob, time
sys.path.insert(0, os.getcwd())
from edithunt.env.grader import grade
from edithunt.env.instance import load_instance
from edithunt.env.tasks import make_env, grade_report
from edithunt.data import CITIES, CITY2STATE, STATES
from edithunt.model import Subject
OUT = sys.argv[1]
S = Subject("Qwen/Qwen2.5-1.5B")
res = {}
def call(env, t, **kw):
    o = json.loads(env.call(t, kw))
    if "error" in o: raise RuntimeError(o["error"])
    return o
def insts(task): return [load_instance(p) for p in sorted(glob.glob(f"results/suite/instances/{task}/*.private.json"))]

# 1. T4 black-box scripted: STATE_Q on k known cities per state, planted only; layer guess fixed
for k in (1, 2):
  for Lg in (10,):
    for inst in insts("T4_detective"):
        env = make_env(S, inst, blackbox=True)
        d = call(env, "describe_task")
        cities = [c for s in STATES for c in d["known_cities"][s][:k]]
        wrong = {}
        for i in range(0, len(cities), 16):
            o = call(env, "run_prompts", template="state_q", cities=cities[i:i+16], answers="states", top_k=1)
            for r in o["results"]:
                ans = r["top_states"][0][0]
                if ans != CITY2STATE[r["city"]]:
                    wrong.setdefault(CITY2STATE[r["city"]], []).append(ans)
        # source = state with most wrong answers (base-rate errors are spread thin)
        src = max(wrong, key=lambda s: len(wrong[s])) if wrong else STATES[0]
        tgt = max(set(wrong.get(src, [STATES[1]])), key=wrong.get(src, [STATES[1]]).count)
        g = grade_report(inst, {"report": {"source": src, "target": tgt, "layer": Lg}})
        res.setdefault(f"T4_bb_statequery_k{k}_L{Lg}", []).append(
            {"id": inst.id, "reward": g["reward"], "comp": g["components"], "calls": env.calls, "n_wrong_states": len(wrong)})
        print("T4", k, inst.id, g["reward"], g["components"], len(wrong), flush=True)

# 2. Fixed-recipe edit: mean-diff on dev fs1 at L=min(Lfix,max_layer), city_last, scale 1, NO evaluation
def fixed_recipe(env, Lfix, scale=1.0, tpl="fs1"):
    d = call(env, "describe_task")
    L = min(Lfix, d["constraints"]["max_layer"])
    call(env, "cache_mean", name="s", cities=d["dev_source_cities"], template=tpl, layer=L)
    call(env, "cache_mean", name="t", cities=d["dev_target_cities"], template=tpl, layer=L)
    call(env, "vec_op", name="d", expr="t - s")
    call(env, "submit", edits=[{"register": "d", "layer": L, "position": "city_last", "scale": scale}])
for task in ("T1_easy", "T3_consistency", "T2_keepstate", "T7_minimal"):
    for Lfix in (8, 12):
        for inst in insts(task):
            env = make_env(S, inst)
            fixed_recipe(env, Lfix)
            g = grade(S, inst, env.submission)
            ex = {k: g[k]["rate"] for k in ("F_cap", "F_state", "F_hidden") if k in g}
            if "flip" in g: ex["flip"] = g["flip"]["rate"]
            if "r" in g: ex["r/rref"] = g["r"] / g["r_ref"]
            res.setdefault(f"{task}_fixed_md_L{Lfix}_s1", []).append({"id": inst.id, "reward": g["reward"], **ex})
            print(task, Lfix, inst.id, round(g["reward"], 3), ex, flush=True)
json.dump(res, open(OUT, "w"), indent=1)

# 3. T7 grid: mean-diff direction x layer x scale (feasibility + best fixed recipe)
for inst in insts("T7_minimal"):
    for L in (4, 6, 7, 8, 10, 12, 14, 16, 18, 21):
        for sc in (0.5, 0.7, 0.85, 1.0, 1.25):
            env = make_env(S, inst)
            fixed_recipe(env, L, sc)
            g = grade(S, inst, env.submission)
            res.setdefault("T7_grid", []).append({"id": inst.id, "L": L, "scale": sc, "reward": g["reward"],
                    "flip": g["flip"]["rate"], "n": g["flip"]["n"], "r": g.get("r"), "r_ref": g.get("r_ref"), "kl": g["kl_mean"]})
        print("T7 grid done", inst.id, flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
json.dump(res, open(OUT, "w"), indent=1)
print("DONE")
