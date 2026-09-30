"""Task difficulty map: per task x agent model, mean reward, pass rate (reward >= 0.5, Wilson CI), turns, cost,
failure categories; plus reference-solver and black-box/scripted markers.
Writes results/task_map.json and results/figures/fig4_task_map.png."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__) + "/..")
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edithunt.common import ROOT
from edithunt.env.instance import load_instance
from edithunt.metrics import rate

TASKS = ["T1_easy", "T2_keepstate", "T3_consistency", "T4_detective", "T6_handoff", "T7_minimal"]
LABEL = {"T1_easy": "T1 easy edit", "T2_keepstate": "T2 keep state", "T3_consistency": "T3 consistency\n(hidden readouts)",
         "T4_detective": "T4 detective", "T6_handoff": "T6 find handoff", "T7_minimal": "T7 minimal edit"}
AGENTS = [("claude-sonnet-5-5", "Claude Sonnet 5.5"), ("gpt-6.1-sol", "GPT-6.1 Sol")]


def failure(r: dict):
    g = r["grade"]
    if g["reward"] >= 0.5:
        return None
    if not g.get("valid"):
        end = r.get("end") or ("cost_cap" if r["stop_reason"] == "cost_cap" else None)  # "end" exists from 2026-09-30
        if g.get("reason") != "no submission":
            return "tool error (invalid submission)"
        return {"cost_cap": "cost cap (no submission)", "max_turns": "turn limit (no submission)"}.get(
            end, "gave up / no submission")
    t = r["suite_task"]
    if t == "T4_detective":
        c = g["components"]
        return "wrong source" if not c["source"] else ("wrong target" if not c["target"] else "wrong layer")
    if t == "T6_handoff":
        return "wrong layer"
    if t == "T7_minimal":
        return ("flip < 0.8" if g["flip"]["rate"] < g.get("min_flip", 0.8) else "side effects") if not g["passed"] \
            else "norm too large"
    if t == "T3_consistency":
        fc, fh = g["F_cap"]["rate"], g["F_hidden"]["rate"]
        if fc >= 0.5 and fh < 0.5:
            return "answer direction (hidden readouts not moved)"
    if g.get("kl_mean", 0) > 0.5 * g.get("kl_budget", 1.0):
        return "side effects (KL)"
    if t == "T2_keepstate" and g.get("state_kept", {}).get("rate", 1) < 0.5:
        return "state moved"
    if g.get("leak_weight", 0) and g.get("leak", {}).get("rate", 0) > 0.5:
        return "leakage"
    return "low flip"


runs = ROOT / "results" / "suite" / "runs"
base = json.loads((ROOT / "results/suite/baselines.json").read_text()) if (ROOT / "results/suite/baselines.json").exists() else {}
M = {"tasks": {}}
for t in TASKS:
    insts = {p.name.replace(".private.json", ""): load_instance(p)
             for p in (ROOT / "results/suite/instances" / t).glob("*.private.json")}
    d = {"n_instances": len(insts),
         "reference_reward": sum(i.extra["reference"]["reward"] or 0 for i in insts.values()) / max(1, len(insts)),
         "scripted": {k.split("/", 1)[1]: sum(x["reward"] for x in v) / len(v) for k, v in base.items() if k.startswith(t + "/")},
         "scripted_pass": {k.split("/", 1)[1]: rate([x["reward"] >= 0.5 for x in v]) for k, v in base.items()
                           if k.startswith(t + "/")},
         "agents": {}}
    for am, _ in AGENTS:
        R = [json.loads(p.read_text()) for p in sorted((runs / am / t).glob("*.json"))] if (runs / am / t).exists() else []
        for bb in (False, True):
            rr = [r for r in R if r.get("blackbox", False) == bb]
            if not rr:
                continue
            rw = [r["grade"]["reward"] for r in rr]
            d["agents"][am + (" (black-box)" if bb else "")] = {
                "n": len(rr), "n_missing": d["n_instances"] - len(rr) if not bb else None, "mean_reward": sum(rw) / len(rw), "pass": rate([x >= 0.5 for x in rw]),
                "mean_turns": sum(r["turns"] for r in rr) / len(rr), "cost_usd": sum(r["cost_usd"] for r in rr),
                "failures": dict(Counter(f for f in map(failure, rr) if f)),
                "ends": dict(Counter(r.get("end") or ("submitted" if r.get("submission") else "?") for r in rr)),
                # T7: the agent is told "passed = F >= 0.8 and KL_mean <= budget"; the map's pass also needs
                # reward >= 0.5 (r <= 2 r_ref), so report the grader's own criterion too
                **({"grader_passed": rate([bool(r["grade"].get("passed")) for r in rr])} if t == "T7_minimal" else {}),
                "episodes": [{"instance": r["instance"], "reward": r["grade"]["reward"], "turns": r["turns"],
                              "cost": r["cost_usd"], "failure": failure(r)} for r in rr]}
    M["tasks"][t] = d
(ROOT / "results/task_map.json").write_text(json.dumps(M, indent=1))

# ---- plot
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"claude-sonnet-5-5": "#2a78d6", "gpt-6.1-sol": "#eb6834"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
fig, ax = plt.subplots(figsize=(8.6, 4.4))
w = 0.3
for j, (am, lab) in enumerate(AGENTS):
    for i, t in enumerate(TASKS):
        a = M["tasks"][t]["agents"].get(am)
        if not a:
            continue
        x = i + (j - 0.5) * w
        p = a["pass"]
        ax.bar(x, a["mean_reward"], width=w - 0.03, color=COL[am], alpha=0.35, edgecolor="none",
               label=f"{lab}: mean reward" if i == 0 else None)
        ax.errorbar(x, p["rate"], yerr=[[p["rate"] - p["ci95"][0]], [p["ci95"][1] - p["rate"]]], fmt="o", ms=6,
                    color=COL[am], mec=SURF, mew=1.5, capsize=3, lw=1.2, label=f"{lab}: pass rate (95% CI)" if i == 0 else None)
        ax.text(x, -0.07, f"{sum(e['reward'] >= 0.5 for e in a['episodes'])}/{a['n']}", ha="center", fontsize=7, color=INK2)
        bb = M["tasks"][t]["agents"].get(am + " (black-box)")
        if bb:
            ax.plot(x, bb["mean_reward"], marker="x", ms=7, color=INK, mew=1.6, ls="none",
                    label="black-box (prompting only), mean" if (j == 0 and t == "T4_detective") else None)
for i, t in enumerate(TASKS):
    ax.plot([i - 0.42, i + 0.42], [M["tasks"][t]["reference_reward"]] * 2, color=INK2, lw=1.2, ls=(0, (3, 2)),
            label="reference solver (via tools), mean" if i == 0 else None)
    sp = M["tasks"][t]["scripted_pass"]
    if sp:  # best zero-/low-effort scripted agent (pass rate), named
        k, v = max(sp.items(), key=lambda kv: kv[1]["rate"])
        ax.plot(i + 0.44, v["rate"], marker="d", ms=6, color="#1baf7a", ls="none",
                label="best scripted baseline: pass rate" if i == 0 else None)
        ax.text(i + 0.44, v["rate"] + 0.045, k.replace("_", " "), ha="center", fontsize=6, color="#127a55")
    g = M["tasks"][t]["agents"].get("claude-sonnet-5-5", {}).get("grader_passed")
    if g:
        ax.plot(i - 0.5 * w, g["rate"], marker="o", ms=6, mfc="none", mec=COL["claude-sonnet-5-5"], mew=1.3, ls="none",
                label="Sonnet 5.5: T7 grader 'passed' (F>=0.8, KL ok)")
ax.axhline(0.5, color=INK2, lw=0.8, alpha=0.5)
ax.set_xticks(range(len(TASKS))); ax.set_xticklabels([LABEL[t] for t in TASKS], fontsize=8.5)
ax.set_ylim(-0.12, 1.08); ax.set_ylabel("reward / pass rate")
ax.set_title("Where frontier agents land on 6 interpretability tasks (Qwen2.5-1.5B subject)", loc="left", fontsize=11)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=7.5)
fig.text(0.01, 0.005, "Bars = mean reward; dots = pass rate (reward >= 0.5) with Wilson 95% CI; n under bars. T6 spans 4 subject models.\n"
         "Scripted: fixed = mean-diff at a fixed layer, no evaluation. T5 (erase) dropped: no reference solver.", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
(ROOT / "results/figures").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "results/figures/fig4_task_map.png", dpi=180)
for t in TASKS:
    for am, a in M["tasks"][t]["agents"].items():
        print(f"{t:15s} {am:32s} n {a['n']:2d} mean {a['mean_reward']:.2f} pass {a['pass']['rate']:.2f} "
              f"[{a['pass']['ci95'][0]:.2f},{a['pass']['ci95'][1]:.2f}] turns {a['mean_turns']:.1f} ${a['cost_usd']:.3f} {a['failures']}")
    print(f"{t:15s} reference {M['tasks'][t]['reference_reward']:.2f} scripted {M['tasks'][t]['scripted']}")
