"""Audit checks for the Phase 7 suite (read-only w.r.t. repo results; writes only to scratchpad)."""
import glob, json, sys
from pathlib import Path
import torch
from edithunt.model import Subject
from edithunt.data import (STATE_Q, STATE_Q_HO, READOUTS, STATES, TEMPLATES, CAPITALS, CITY2STATE)
from edithunt.env.instance import load_instance
from edithunt.env.grader import grade, check, build_ivs, predict, CAPS
from edithunt.env import tasks as T
from edithunt.env.tools import ToolEnv

OUT = Path(sys.argv[1])
S = Subject("Qwen/Qwen2.5-1.5B")
res = {}

def edits_of(inst, sub):
    e, why = check(sub, inst.constraints, S.d_model, S.n_layers)
    return e

def kept_on(inst, edits, cities):
    out = {}
    for tag, tpl, cands, ans in [("state_q", STATE_Q, STATES, None), ("state_q_ho", STATE_Q_HO, STATES, None),
                                 ("abbr", READOUTS["abbr"][0], [READOUTS["abbr"][1][s] for s in STATES], READOUTS["abbr"][1])]:
        e = [S.encode(tpl, c) for c in cities]
        p0, p1 = predict(S, e, cands, ()), predict(S, e, cands, build_ivs(edits, e))
        src = ans[inst.source] if ans else inst.source
        # among cities the clean model answers with the source state on this wording
        ok = [b == a for a, b in zip(p0, p1) if a == src]
        out[tag] = (sum(ok), len(ok))
    return out

def ext_F(inst, edits, tkeys=("ho_fs", "ho_zs", "fs2", "zs2")):
    """Flip rate on test cities x more wordings (clean-valid items only)."""
    cities = sorted({c for c, _ in inst.test_items})
    f = []
    for tk in tkeys:
        e = [S.encode(TEMPLATES[tk], c) for c in cities]
        p0 = predict(S, e, CAPS, ())
        p1 = predict(S, e, CAPS, build_ivs(edits, e))
        f += [b == inst.target_capital for a, b, c in zip(p0, p1, cities) if a == CAPITALS[CITY2STATE[c]]]
    return (sum(f), len(f))

subs = {}
INST = {}
for ip in glob.glob("results/suite/instances/*/*.private.json"):
    d = json.load(open(ip))
    if d["model"] == S.name and d["suite_task"] in ("T1_easy", "T2_keepstate", "T3_consistency", "T7_minimal"):
        INST[d["id"]] = ip
replay = []
for p in sorted(glob.glob("results/suite/runs/*/*/*.json")):
    r = json.load(open(p))
    who = p.split("/")[3][:6]
    if who.startswith("dry") or r["instance"] not in INST or not r.get("submission_edits"):
        continue
    inst = load_instance(INST[r["instance"]])
    env = T.make_env(S, inst)
    for x in r["tool_log"]:
        env.call(x["tool"], x["args"])
    g = grade(S, inst, env.submission) if env.submission else {"reward": None}
    replay.append({"run": p, "recorded": r["grade"].get("reward"), "replayed": g.get("reward")})
    print("replay", r["instance"], who, "recorded", r["grade"].get("reward"), "replayed", g.get("reward"), flush=True)
    if env.submission:
        subs.setdefault(r["instance"], []).append((who, env.submission, g.get("reward")))
res["replay"] = replay

# ---- 1+2: T2
t2 = {}
for ip in sorted(glob.glob("results/suite/instances/T2_keepstate/*.private.json")):
    inst = load_instance(ip)
    cities = sorted({c for c, _ in inst.test_items})
    row = {"refs": {}, "subs": []}
    for seed in (0, 1, 2, 3):
        r = T.run_reference(S, inst, seed=seed)
        g = r["grade"]
        ed = edits_of(inst, r["submission"]) if r["submission"] else None
        row["refs"][seed] = {"reward": g.get("reward"), "F": g.get("F"), "leak": g.get("leak", {}).get("rate"),
                             "kl": g.get("kl_mean"), "kept": kept_on(inst, ed, cities) if ed else None,
                             "extF": ext_F(inst, ed) if ed else None}
        print(inst.id, "ref seed", seed, row["refs"][seed], flush=True)
    for who, sub, rw in subs.get(inst.id, []):
        ed = edits_of(inst, sub)
        row["subs"].append({"who": who, "reward": rw, "kept": kept_on(inst, ed, cities), "extF": ext_F(inst, ed)})
        print(inst.id, who, row["subs"][-1], flush=True)
    t2[inst.id] = row
res["T2"] = t2

# ---- 3: extended-item re-score of T1/T3/T7 submissions
ext = {}
for task in ("T1_easy", "T3_consistency", "T7_minimal"):
    for ip in sorted(glob.glob(f"results/suite/instances/{task}/*.private.json")):
        inst = load_instance(ip)
        for who, sub, rw in subs.get(inst.id, []):
            ed = edits_of(inst, sub)
            ext.setdefault(task, []).append({"id": inst.id, "who": who, "reward": rw, "extF": ext_F(inst, ed)})
            print(task, inst.id, who, rw, ext[task][-1]["extF"], flush=True)
res["ext"] = ext

# ---- 4: refusal message as a membership oracle for private cities
inst = load_instance(sorted(glob.glob("results/suite/instances/T2_keepstate/*.private.json"))[0])
env = ToolEnv(S, inst)
leak_c = sorted({c for c, _ in inst.leak_items})
probe = leak_c[:2] + inst.test_cities[:1] + ["Denver"]
res["oracle"] = {"probe": probe, "reply": json.loads(env.call("run_prompts", {"template": "fs1", "cities": probe})),
                 "truth_leak": leak_c, "truth_test": inst.test_cities}
print(res["oracle"])
OUT.write_text(json.dumps(res, indent=1, default=str))
