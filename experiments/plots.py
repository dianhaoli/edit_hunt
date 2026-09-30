"""Plots from saved result JSONs. Usage: plots.py --model Qwen/Qwen2.5-1.5B"""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__) + "/..")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edithunt.common import base_args, load, rdir

args = base_args(__doc__).parse_args()
out = rdir(args.model)
short = args.model.split("/")[-1]


def curve(ax, agg, key, label, color, ls="-", metric="flip"):
    if key not in agg:
        return
    Ls = sorted(int(L) for L in agg[key])
    r = [agg[key][str(L)][metric] for L in Ls]
    y = [x["rate"] for x in r]
    lo = [x["ci95"][0] for x in r]; hi = [x["ci95"][1] for x in r]
    ax.plot(Ls, y, ls, color=color, label=label, lw=2, marker="o", ms=3)
    ax.fill_between(Ls, lo, hi, color=color, alpha=0.15, lw=0)


try:
    p1 = load(args.model, "phase1_patching")
    agg = p1["result"]["agg"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    ax = axes[0]
    curve(ax, agg, "A_city|fs1", "A: full-resid patch @ city token", "#1f77b4")
    curve(ax, agg, "A_final|fs1", "A: full-resid patch @ final token", "#d62728")
    ax.set_title(f"{short}: full residual patch (fs1)\nflip rate to target capital, 95% Wilson CI")
    ax.set_xlabel("layer"); ax.set_ylabel("flip rate"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axes[1]
    curve(ax, agg, "B_meandiff|fs1", "mean-diff, build template (fs1)", "#2ca02c")
    curve(ax, agg, "B_meandiff|ho_fs", "mean-diff, held-out template (ho_fs)", "#9467bd")
    curve(ax, agg, "B_meandiff|ho_zs", "mean-diff, held-out zero-shot (ho_zs)", "#8c564b", "--")
    curve(ax, agg, "ctrl_random|fs1", "random norm-matched", "#7f7f7f", ":")
    curve(ax, agg, "ctrl_shuffled|fs1", "shuffled-label mean-diff", "#bcbd22", ":")
    ax.set_title(f"{short}: shared state direction added @ city token\nheld-out source cities, all pairs pooled")
    ax.set_xlabel("layer"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "phase1_layers.png", dpi=140)
    print("wrote", out / "phase1_layers.png")
except FileNotFoundError:
    pass
