"""Are T8 agent failures interpretability-skill gaps or environment issues? (user request, 2026-10-02; rule fixed in
LAB_NOTEBOOK before this ran). Per agent episode:
 1. replay the tool log through a fresh ToolEnv (registers rebuilt; error pattern compared with the record)
 2. re-grade the recorded submission (grader consistency)
 3. counterfactual: grade every distinct edit set the agent evaluated with eval_intervention (its candidates), esp.
    the last ones before the episode ended
 4. mechanism diagnosis of each candidate / the submission: layer vs plant layers, cos(scale * vector, -sum v_true),
    gate opening on affected eval cities vs other eval cities (post-plant)
Classification: ENV_GRADER if |regrade - recorded| > 0.02; ENV_TURNCAP if the episode hit max_turns and its best
submittable evaluated candidate scores >= 0.8 x solver; else SKILL with sub-labels wrong_layer (outside
[min plant layer, max plant layer + 2]), wrong_direction (|cos| < 0.5), wrong_gate (opens < 0.5 on affected or > 0.3
on others). Internal (OOM) errors and refusals are reported alongside.
  PYTHONPATH=. .venv/bin/python experiments/t8_failure_audit.py [record.json ...]"""
import copy
import glob
import json
import sys

import torch

from edithunt.common import ROOT
from edithunt.data import TEMPLATES
from edithunt.env import undo as U
from edithunt.env.grader import check
from edithunt.env.instance import load_instance
from edithunt.env.tasks import make_env
from edithunt.model import Subject, pos_rule

S = Subject("Qwen/Qwen2.5-1.5B")
pilot = {}
for pf in ("pilot.jsonl", "pilot_v02.jsonl"):
    for l in (ROOT / "results/suite/t8_undo" / pf).read_text().splitlines():
        r = json.loads(l)
        pilot.setdefault(r["id"], {})[r["method"]] = r["R"]


def to_sub(env, edits):
    """Agent edit list (register names) -> submission dict (as t_submit builds it), or None if not submittable."""
    es = []
    for e in edits:
        if "register" not in e or e["register"] not in env.regs:
            return None
        d = {"layer": e["layer"], "position": e.get("position", "city_last"),
             "vector": env.regs[e["register"]].tolist(), "scale": float(e.get("scale", 1.0))}
        sf = e.get("scale_from")
        if sf:
            if sf.get("key") not in env.regs:
                return None
            d["scale_from"] = {"key": env.regs[sf["key"]].tolist(), "lo": float(sf["lo"]), "hi": float(sf["hi"])}
        es.append(d)
    sub = {"kind": "add", "edits": es}
    ok, why = check(sub, env.inst.constraints, S.d_model, S.n_layers)
    return sub if ok is not None else None


def diagnose(inst, sub):
    gates = inst.extra["plant"]["gates"]
    vtrue = -sum(torch.tensor(g["vector"]) for g in gates)
    Ls = [g["layer"] for g in gates]
    out = []
    for e in sub["edits"]:
        v = torch.tensor(e["vector"]) * e["scale"]
        d = {"layer": e["layer"], "position": e["position"], "layer_ok": min(Ls) <= e["layer"] <= max(Ls) + 2,
             "cos_minus_v": torch.nn.functional.cosine_similarity(v, vtrue, dim=0).item(), "norm_ratio": (v.norm() / vtrue.norm()).item()}
        sf = e.get("scale_from")
        if sf:
            k = torch.tensor(sf["key"])
            def opening(cities):
                es = U._enc(S, TEMPLATES["fs1"], cities)
                if not es:
                    return None
                _, st, _, _ = S.forward(es, U.plant_ivs(inst, es), {e["layer"]: pos_rule(es, "city_last")})
                h = st.captures[e["layer"]][:, 0].float().cpu()
                return ((h @ k - sf["lo"]) / (sf["hi"] - sf["lo"])).clamp(0, 1).mean().item()
            d["gate_affected"] = opening(inst.extra["eval_src"][:16])
            d["gate_others"] = opening(inst.extra["eval_oth"][:16])
        out.append(d)
    return out


def edits_of(args):
    if args.get("edits"):
        return args["edits"]
    if args.get("vector"):
        e = {"register": args["vector"], "layer": args.get("layer"), "position": args.get("position", "city_last"),
             "scale": args.get("scale", 1.0)}
        return [e]
    return None


def main():
    paths = sys.argv[1:] or sorted(glob.glob(str(ROOT / "results/suite/runs/*/T8_undo/*.json")))
    if paths == ["v02"]:
        paths = sorted(glob.glob(str(ROOT / "results/suite/runs/*/T8_undo_v02/*.json")))
    out = ROOT / "results/suite/t8_undo/failure_audit.jsonl"
    f = out.open("a")
    for p in paths:
        rec = json.loads(open(p).read())
        inst = load_instance(glob.glob(str(ROOT / f"results/suite/instances/T8_undo*/{rec['instance']}.private.json"))[0])
        sol = pilot[inst.id]["solver"]
        env = make_env(S, inst)
        rec_outs = [c.get("output") or c.get("content") or "" for m in rec["messages"] if m["role"] == "user"
                    and isinstance(m["content"], list) for c in m["content"]]
        mism, cands = 0, []
        for i, x in enumerate(rec["tool_log"]):
            if x["tool"] == "submit":
                break
            o = env.call(x["tool"], x["args"])
            if i >= 1 and i - 1 < len(rec_outs) and o.startswith('{"error"') != rec_outs[i - 1].startswith('{"error"'):
                mism += 1
            if x["tool"] == "eval_intervention" and not o.startswith('{"error"'):
                ed = edits_of(x["args"])
                if ed:
                    snap = {k: v.clone() for k, v in env.regs.items()}
                    cands.append((i, copy.deepcopy(ed), snap))
        regrade = None
        if rec.get("submission_full"):
            regrade = U.grade_undo(S, inst, rec["submission_full"])["reward"]
        # distinct submittable candidates, last 6
        seen, graded = set(), []
        for i, ed, snap in reversed(cands):
            key = json.dumps(ed, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            env.regs = snap
            sub = to_sub(env, ed)
            if sub is None:
                graded.append({"call": i, "edits": ed, "submittable": False})
            else:
                g = U.grade_undo(S, inst, sub)
                graded.append({"call": i, "edits": [{k: v for k, v in e.items() if k != "scale_from"} | (
                    {"scale_from": {kk: vv for kk, vv in e["scale_from"].items() if kk != "key"}} if e.get("scale_from") else {})
                    for e in ed], "submittable": True, "R": g["reward"], "diag": diagnose(inst, sub)})
            if len(graded) >= 6:
                break
        best = max([c["R"] for c in graded if c.get("submittable")], default=None)
        n_int = sum("internal error" in o for o in rec_outs)
        n_ref = sum('"not accessible"' in o for o in rec_outs)
        R = rec["grade"].get("reward", 0.0)
        diag_sub = diagnose(inst, rec["submission_full"]) if rec.get("submission_full") else None
        if regrade is not None and abs(regrade - R) > 0.02:
            cls = "ENV_GRADER"
        elif R >= 0.8 * sol:
            cls = "PASS"
        elif rec["end"] in ("max_turns", "cost_cap") and best is not None and best >= 0.8 * sol:
            cls = "ENV_TURNCAP" if rec["end"] == "max_turns" else "ENV_COSTCAP"
        else:
            d = diag_sub or (max((c for c in graded if c.get("submittable")), key=lambda c: c["R"])["diag"]
                             if any(c.get("submittable") for c in graded) else None)
            subs = []
            if d:
                e0 = d[0]
                if not e0["layer_ok"]:
                    subs.append("wrong_layer")
                if abs(e0["cos_minus_v"]) < 0.5:
                    subs.append("wrong_direction")
                if e0.get("gate_affected") is not None and (e0["gate_affected"] < 0.5 or (e0.get("gate_others") or 0) > 0.3):
                    subs.append("wrong_gate")
            cls = "SKILL" + (":" + ",".join(subs) if subs else ":no_candidate" if not d else ":other")
        row = {"record": p.split("/")[-1], "instance": inst.id, "model": rec["agent_model"],
               "name": inst.extra.get("name", inst.id), "plant_layers": [g["layer"] for g in inst.extra["plant"]["gates"]],
               "solver": sol, "recorded_R": R, "regraded_R": regrade, "end": rec["end"], "turns": rec["turns"],
               "replay_error_mismatches": mism, "internal_errors": n_int, "refusals": n_ref,
               "n_candidates": len(cands), "best_candidate_R": best, "candidates": graded, "submission_diag": diag_sub,
               "class": cls}
        f.write(json.dumps(row) + "\n"); f.flush()
        print(row["record"], cls, "R %.3f regrade %s best cand %s solver %.3f internal %d refusals %d mismatches %d" % (
            R, None if regrade is None else round(regrade, 3), None if best is None else round(best, 3), sol, n_int, n_ref, mism),
            flush=True)


if __name__ == "__main__":
    main()
