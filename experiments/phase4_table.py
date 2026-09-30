"""Phase 4 ladder table: per (class, layer, ndev, pos), pooled over pairs.
Reads phase4_ladder.json (or the per-pair checkpoint with --partial).
  python experiments/phase4_table.py --model Qwen/Qwen2.5-1.5B [--partial] [--ndev all]"""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__) + "/..")
import argparse
import json

from edithunt.common import rdir
from edithunt.metrics import rate

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--partial", action="store_true")
ap.add_argument("--ndev", default="all", help="ndev label to show for C1/C1t (1, 2, all); others are always 'all'")
a = ap.parse_args()
d = rdir(a.model)
if a.partial:
    rows = json.loads((d / "phase4_ladder.partial.json").read_text())["rows"]
else:
    rows = json.loads((d / "phase4_ladder.json").read_text())["result"]["rows"]
pairs = sorted({r["pair"] for r in rows})
print(f"{len(pairs)} pairs")


def pool(R, k):
    return rate([f for r in R for f in r.get(k, [])])


def mean(R, k):
    xs = [x for r in R for x in r.get(k, [])]
    return sum(xs) / len(xs) if xs else float("nan")


keys = sorted({(r["cls"], r["L"], str(r["ndev"]), r["pos"]) for r in rows}, key=lambda k: (k[0], k[1], k[2], k[3]))
hdr = "cls      L  ndev pos        flip_fs1          ho_fs  ho_zs  state>t 3rd>tgt 3rd_kept binKLc  KLgen   |v|"
print(hdr)
for k in keys:
    if k[0] in ("C1", "C1t") and k[2] != a.ndev:
        continue
    R = [r for r in rows if (r["cls"], r["L"], str(r["ndev"]), r["pos"]) == k]
    f = pool(R, "flips_fs1")
    if k[0] == "C4":
        print(f"{k[0]:7s} {'-':>3s} {k[2]:>5s} {k[3]:10s} {f['rate']:.2f} (n={f['n']:3d})       "
              f"{pool(R, 'flips_ho_fs')['rate']:.2f}   {pool(R, 'flips_ho_zs')['rate']:.2f}")
        continue
    ns = [r["norm"] for r in R if r.get("norm") is not None]
    print(f"{k[0]:7s} {k[1]:3d} {k[2]:>5s} {k[3]:10s} {f['rate']:.2f} [{f['ci95'][0]:.2f},{f['ci95'][1]:.2f}] "
          f"(n={f['n']:3d}) {pool(R, 'flips_ho_fs')['rate']:.2f}   {pool(R, 'flips_ho_zs')['rate']:.2f}   "
          f"{pool(R, 'state_to_tgt')['rate']:.2f}    {pool(R, 'third_to_tgt')['rate']:.2f}    "
          f"{pool(R, 'third_kept')['rate']:.2f}     {mean(R, 'kl_country'):.3f}  {mean(R, 'kl_generic'):.3f}  "
          f"{(sum(ns) / len(ns)) if ns else float('nan'):6.1f}")
