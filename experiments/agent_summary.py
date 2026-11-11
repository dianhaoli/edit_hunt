"""Summary of v2.1 / T8 agent episodes against the pre-registered bars (LAB_NOTEBOOK 2026-10-02).
T2 v2.1: pass = R >= 0.8 x mean of the 3 v2.1 reference seeds (results/suite/t2v21/library_summary.json).
T8: pass = restoration >= 0.8 x solver (results/suite/t8_undo/pilot.jsonl).
Episodes whose tools returned an 'internal error' (e.g. CUDA OOM on the shared GPU) are flagged as tainted.
  PYTHONPATH=. .venv/bin/python experiments/agent_summary.py"""
import json
import re
import statistics as st
from collections import defaultdict

from edithunt.common import ROOT

RUNS = ROOT / "results" / "suite" / "runs"
lib = json.loads((ROOT / "results/suite/t2v21/library_summary.json").read_text())["instances"]
pilot = defaultdict(dict)
for pp in (ROOT / "results/suite/t8_undo/pilot.jsonl", ROOT / "results/suite/t8_undo/pilot_v02.jsonl"):
    if pp.exists():
        for l in pp.read_text().splitlines():
            r = json.loads(l)
            pilot[r["id"]][r["method"]] = r["R"]


def tainted(rec) -> int:
    n = 0
    for m in rec["messages"]:
        if m["role"] == "user" and isinstance(m["content"], list):
            for c in m["content"]:
                txt = c.get("content") or c.get("output") or ""
                if isinstance(txt, str) and "internal error" in txt:
                    n += 1
    return n


out = {}
for task in ("T2_keepstate_v21", "T8_undo", "T8_undo_v02"):
    rows = []
    for p in sorted(RUNS.glob(f"*/{task}/*.json")):
        rec = json.loads(p.read_text())
        tag = (re.sub(r"^.*?-s0(-v21)?\.?", "", p.stem) if not p.stem.startswith("T8u-") else
               (p.stem.split(".", 1)[1] if "." in p.stem else "")) or "r0"
        iid = rec["instance"]
        R = rec["grade"].get("reward", 0.0)
        if task == "T2_keepstate_v21":
            ref = lib[iid]["ref_mean"]
            bar, extra = 0.8 * ref, {"ref": ref, "best_script": lib[iid]["best_R"]}
        else:
            sol = pilot[iid].get("solver")
            bar = 0.8 * sol if sol else None
            extra = {"solver": sol, "best_script": max((v for k, v in pilot[iid].items() if k != "solver"), default=None)}
        gated = any("scale_from" in e for e in rec.get("submission_edits", []))
        rows.append({"model": rec["agent_model"], "tag": tag, "id": iid, "R": R, "bar": bar,
                     "pass": bar is not None and R >= bar, "end": rec.get("end"), "cost": rec["cost_usd"],
                     "turns": rec["turns"], "tainted": tainted(rec), "gated": gated,
                     "positions": [e.get("position") for e in rec.get("submission_edits", [])],
                     "layers": [e.get("layer") for e in rec.get("submission_edits", [])], **extra})
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["tag"].split(".")[0] if r["tag"] else "r0")].append(r)
    summ = {}
    for (m, tag), rs in sorted(groups.items()):
        clean = [r for r in rs if not r["tainted"]]
        summ[f"{m}/{tag}"] = {"n": len(rs), "n_tainted": len(rs) - len(clean), "pass": sum(r["pass"] for r in clean),
                              "n_clean": len(clean), "mean_R": st.mean(r["R"] for r in clean) if clean else None,
                              "cost": round(sum(r["cost"] for r in rs), 3),
                              "ends": dict(sorted({e: sum(r["end"] == e for r in rs) for e in {r["end"] for r in rs}}.items()))}
    # pass@k per instance over runs of the same model with tags r1, r2, ...
    by_inst = defaultdict(list)
    for r in rows:
        if re.fullmatch(r"r\d+", r["tag"] or "") and not r["tainted"]:
            by_inst[(r["model"], r["id"])].append(r)
    passk = {f"{m}|{i}": {"runs": len(v), "passes": sum(x["pass"] for x in v), "R": [round(x["R"], 3) for x in v]}
             for (m, i), v in sorted(by_inst.items())}
    out[task] = {"groups": summ, "pass_at_k": passk, "rows": rows}
    print(f"== {task}")
    for k, v in summ.items():
        print(f"  {k:28s} n {v['n']} (tainted {v['n_tainted']}) pass {v['pass']}/{v['n_clean']} mean R "
              f"{v['mean_R'] if v['mean_R'] is None else round(v['mean_R'], 3)} cost ${v['cost']} ends {v['ends']}")
(ROOT / "results/suite/t2v21/agent_summary.json").write_text(json.dumps(out, indent=1, default=str))
