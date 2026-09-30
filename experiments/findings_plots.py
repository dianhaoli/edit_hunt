"""Three figures for FINDINGS.md (written to results/figures/).
  fig1_layer_curves.png   mean-diff held-out flip rate vs normalized depth, 4 models, handoff marked
  fig2_ladder.png         Phase 4/4b: flip (held-out template) vs third-state leakage, per intervention class
  fig3_tiers.png          Phase 6 scripted-agent calibration: mean reward per tier (dots = instances)
Palette: dataviz reference categorical slots (fixed order), light surface."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__) + "/..")
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edithunt.common import ROOT

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def load(model, name):
    return json.loads((ROOT / "results" / model / f"{name}.json").read_text())["result"]


# ---------------------------------------------------------------- fig 1
MODELS = [("Qwen2.5-1.5B", 22, S1), ("Qwen2.5-3B", 31, S2), ("gemma-2-2b", 18, S3), ("Qwen2.5-7B", 22, S4)]
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for m, handoff, col in MODELS:
    a = load(m, "phase1_patching")
    a = a.get("agg") or a
    cur = a["B_meandiff|fs1"]
    Ls = sorted(int(k) for k in cur)
    n = max(Ls) + 1
    xs = [L / n for L in Ls]
    ys = [cur[str(L)]["flip"]["rate"] for L in Ls]
    ax.plot(xs, ys, color=col, lw=2, label=f"{m} ({n} layers)", solid_capstyle="round")
    ax.plot([handoff / n], [cur[str(handoff)]["flip"]["rate"]], "o", ms=8, color=col, mec=SURF, mew=2, zorder=3)
rnd = load("Qwen2.5-1.5B", "phase1_patching")
rnd = (rnd.get("agg") or rnd)["ctrl_random|fs1"]
Lr = sorted(int(k) for k in rnd)
ax.plot([L / 28 for L in Lr], [rnd[str(L)]["flip"]["rate"] for L in Lr], color=INK2, lw=1.2, ls=(0, (3, 3)),
        label="random vector, same norm (1.5B)")
ax.set_xlabel("layer / depth (edit added at the city's last token)")
ax.set_ylabel("held-out cities flipped to target capital")
ax.set_ylim(-0.02, 1.02)
ax.set_title("A shared state direction exists in all 4 models, until the handoff (●)", loc="left", fontsize=11)
ax.legend(loc="center left", fontsize=8.5)
fig.text(0.01, 0.005, "Mean-diff vector (target − source dev cities), few-shot template, 30 pairs, n≈105 held-out cities "
         "per point.\n● = handoff layer (final-token patching takes over).", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(OUT / "fig1_layer_curves.png", dpi=180)
plt.close(fig)

# ---------------------------------------------------------------- fig 2
p4 = json.loads((ROOT / "results/Qwen2.5-1.5B/phase4_ladder.json").read_text())["result"]["rows"]
p4b = json.loads((ROOT / "results/Qwen2.5-1.5B/phase4b_careful.json").read_text())["result"]["rows"]


def pooled(rows, cls, L, k, ndev="all"):
    xs = [f for r in rows if r["cls"] == cls and r["L"] == L and str(r.get("ndev", "all")) == ndev
          and r.get("pos", "city_last") == "city_last" for f in r.get(k, [])]
    return sum(xs) / len(xs) if xs else None


# (label, rows, cls, flip key, leak key, family color, marker)
CLASSES = [("full paste (C0)", p4, "C0", "flips_ho_fs", "third_to_tgt", S2, "D", "1"),
           ("gradient, uncapped (C2)", p4, "C2", "flips_ho_fs", "third_to_tgt", S2, "^", "all"),
           ("gradient, norm-capped (C2n)", p4, "C2n", "flips_ho_fs", "third_to_tgt", S2, "v", "all"),
           ("mean-diff (C1)", p4, "C1", "flips_ho_fs", "third_to_tgt", S1, "o", "all"),
           ("DAS rank-1 (C3)", p4, "C3r1", "flips_ho_fs", "third_to_tgt", S1, "s", "all"),
           ("gradient + keep other states (C2nk)", p4b, "C2nk", "flips_ho_fs", "third_to_tgt_ho_fs", S3, "*", "all")]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for lab, rows, cls, fk, lk, col, mk, nd in CLASSES:
    pts = []
    for L in (4, 5, 8, 15):
        f, l = pooled(rows, cls, L, fk, nd), pooled(rows, cls, L, lk, nd)
        if f is not None and l is not None:
            pts.append((l, f, L))
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=200 if mk == "*" else 55, marker=mk, color=col,
               edgecolor=SURF, linewidth=1.5, label=lab, zorder=3)
    if cls in ("C2nk", "C1", "C2n", "C3r1"):
        for l, f, L in pts:
            ax.annotate(f"L{L}", (l, f), xytext=(5, -3), textcoords="offset points", fontsize=7, color=INK2)
ax.axvspan(-0.02, 0.15, color=S3, alpha=0.06, zorder=0)
ax.text(0.005, 0.03, "low leakage", fontsize=8, color=INK2)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.05)
ax.set_xlabel("leakage: other states' cities that switch to the target capital")
ax.set_ylabel("held-out cities flipped (unseen few-shot template)")
ax.set_title("Only a leak-aware method is both effective and specific (Qwen2.5-1.5B)", loc="left", fontsize=11)
ax.legend(loc="lower right", fontsize=8)
fig.text(0.01, 0.005, "12 pairs; n=42 source cities, ≈70 third-state cities per point; layers 4/5/8/15.\n"
         "Orange = answer-direction edits, blue = state-variable edits, green = leak-aware.", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(OUT / "fig2_ladder.png", dpi=180)
plt.close(fig)

# ---------------------------------------------------------------- fig 3
base = load("Qwen2.5-1.5B", "phase6_baselines")["rows"]
v2 = load("Qwen2.5-1.5B", "phase6_careful_v2")["rows"]
# careful = best scripted recipe per tier: v1 (base run) on easy/medium, v2 (+keep_state) on hard
rows = [r for r in base if not (r["agent"] == "careful" and r["tier"] == "hard")] + \
       [r | {"tier": "hard"} for r in v2 if r["instance"].startswith("hard")]
AG = [("meandiff", "mean-diff", S1), ("gradient", "naive gradient", S2), ("careful", "careful (scripted)", S3),
      ("random", "random", S4)]
TIERS = ["easy", "medium", "hard"]
fig, ax = plt.subplots(figsize=(7.2, 4.0))
w = 0.2
for j, (ag, lab, col) in enumerate(AG):
    for i, t in enumerate(TIERS):
        rw = [r["reward"] for r in rows if r["agent"] == ag and r["tier"] == t]
        if not rw:
            continue
        x = i + (j - 1.5) * w
        m = sum(rw) / len(rw)
        ax.bar(x, m, width=w - 0.02, color=col, alpha=0.35, edgecolor="none", label=lab if i == 0 else None)
        ax.scatter([x] * len(rw), rw, s=14, color=col, edgecolor=SURF, linewidth=0.8, zorder=3)
        k = sum(v >= 0.5 for v in rw)
        ax.text(x, max(m, max(rw)) + 0.03, f"{k}/{len(rw)}", ha="center", fontsize=7.5, color=INK2)
ax.axhline(0.5, color=INK2, lw=1, ls=(0, (3, 3)))
ax.text(2.55, 0.51, "pass", fontsize=8, color=INK2, va="bottom", ha="right")
ax.set_xticks(range(3)); ax.set_xticklabels(["easy\n(no leak penalty)", "medium\n(leak penalty, layer ≤15)",
                                             "hard\n(keep state, layer ≤8)"])
ax.set_ylim(0, 1.12); ax.set_ylabel("reward"); ax.grid(axis="x", visible=False)
ax.set_title("Naive methods pass easy and fail medium/hard; a careful recipe passes some", loc="left", fontsize=11)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8, ncol=4)
fig.text(0.01, 0.005, "Qwen2.5-1.5B, 7-8 validated instances per tier; bar = mean reward, dots = instances, label = "
         "passes (reward ≥ 0.5).\nCareful: v1 on easy/medium, v2 (+keep_state) on hard.", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(OUT / "fig3_tiers.png", dpi=180)
plt.close(fig)
print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
