"""Re-grade every existing T2 v2 shortcut / oracle run under the fixed country-KL term (old vs new).

Rows did not store edit vectors, so each edit is regenerated with the same deterministic recipe + seed; the
regenerated grade's Cause/Iso are checked against the stored row (reproducibility), and edits are saved to
results/suite/t2_ravel/edits/ for future regrades. Output: results/suite/t2_ravel/regrade.jsonl
  PYTHONPATH=. .venv/bin/python experiments/t2_regrade.py --shard 0/2
"""
import argparse
import json
import time

from edithunt.common import ROOT
from edithunt.model import Subject
from edithunt.env import ravel as R

D = ROOT / "results" / "suite" / "t2_ravel"
ADDITIVE = {  # experiments/t2_oracle_search.py
    "cap2": dict(cap=2.0), "cap4": dict(cap=4.0), "uncapped": dict(cap=None),
    "cap2_allkeep": dict(cap=2.0, all_keep=True),
    "cap2_allkeep_stx": dict(cap=2.0, all_keep=True, state_extra=True),
    "cap4_allkeep_stx_w3": dict(cap=4.0, all_keep=True, state_extra=True, w_state=3.0, w_other=3.0),
    "cap4_allkeep_stx_w3_300": dict(cap=4.0, all_keep=True, state_extra=True, w_state=3.0, w_other=3.0, steps=300),
}
GATED = {  # experiments/t2_gated.py, t2_gated_boost.py
    "gated": dict(), "gated_stx": dict(state_extra=True), "gated_light300": dict(steps=300, w_keep=0.3),
    "gated_boost": dict(steps=300, templates=["fs1", "zs1", "fs2", "zs2"]),
}


def rows():
    out = []
    for f in sorted(D.glob("shortcut.*.jsonl")) + sorted(D.glob("oracle.*.jsonl")):
        out += [dict(json.loads(l), src=f.name, variant=json.loads(l)["method"]) for l in f.read_text().splitlines()]
    for f in sorted(D.glob("oracle_search.*.jsonl")):
        out += [dict(json.loads(l), src=f.name, method="oracle_search", seed=0) for l in f.read_text().splitlines()]
    return out


def edits_for(S, inst, L, r):
    v, sd = r["variant"], r.get("seed", 0)
    md = R.meandiff(S, inst, L)
    nrm = md.norm().item()
    if v == "null":
        return [(L, "city_last", md * 0)]
    if v == "meandiff":
        return [(L, "city_last", md)]
    if v == "random":
        return R.random_vec(S, nrm, L, sd)
    if v in ("naive_capped", "naive_uncapped"):
        return R.train_vector(S, inst, L, sd, 150, keep=False, max_norm=nrm if v == "naive_capped" else None, lr=0.05)
    if v == "oracle":
        return R.train_vector(S, inst, L, sd, 150, keep=True, max_norm=nrm, lr=0.05)
    if v == "das":
        return R.train_vector(S, inst, L, sd, 150, keep=True, das=True, lr=0.05)
    if v in ADDITIVE:
        kw = dict(ADDITIVE[v]); cap = kw.pop("cap")
        return R.train_vector(S, inst, L, 0, kw.pop("steps", 150), keep=True, max_norm=cap * nrm if cap else None, **kw)
    if v in GATED:
        kw = dict(GATED[v])
        return R.train_gated(S, inst, L, 0, kw.pop("steps", 150), **kw)
    raise ValueError(v)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
    k, n = map(int, a.shard.split("/"))
    S = Subject("Qwen/Qwen2.5-1.5B")
    (D / "edits").mkdir(exist_ok=True)
    out = D / f"regrade.{k}.jsonl"
    done = {(r["src"], r["pair"], r["layer"], r["variant"], r["seed"]) for r in map(json.loads, out.read_text().splitlines())} \
        if out.exists() else set()
    f = out.open("a")
    Ps = {}
    for i, r in enumerate(rows()):
        key = (r["src"], r["pair"], r["layer"], r["variant"], r.get("seed", 0))
        if i % n != k or key in done:
            continue
        src, tgt = r["pair"].split("->")
        P = Ps.setdefault(r["pair"], R.make_pair(S, src, tgt))
        inst = R.make_instance(S, P, r["layer"])
        t = time.time()
        e = edits_for(S, inst, r["layer"], r)
        R.save_edits(D / "edits" / ("%s-%s-L%d-%s-s%d.pt" % (r["src"].split(".")[0], r["pair"].replace(" ", "_").replace("->", "_to_"),
                                                            r["layer"], r["variant"], r.get("seed", 0))), e)
        g = R.grade_ravel(S, inst, None, edits=e)
        kf_old = 1 - min(1, g["kl_mean_oldprobe"])
        C, Is, Io = (g[x]["rate"] for x in ("Cause", "Iso_state", "Iso_other"))
        o = {"src": r["src"], "pair": r["pair"], "layer": r["layer"], "variant": r["variant"], "seed": r.get("seed", 0),
             "Cause": C, "Iso_state": Is, "Iso_other": Io,
             "repro": {x: round(g[x]["rate"] - r[x], 3) for x in ("Cause", "Iso_state", "Iso_other")},
             "old": {"kl_mean": g["kl_mean_oldprobe"], "reward": g["Disentangle"] * kf_old,
                     "reward_mult": C * Is * Io * kf_old, "stored_reward": r["reward"]},
             "new": {"kl_mean": g["kl_mean"], "kl_country": g["kl_country_fixed"]["mean"],
                     "kl_generic": g["kl_generic"]["mean"], "reward": g["reward"], "reward_mult": g["reward_mult"],
                     "reward_product": g["reward_product"], "excluded": g["kl_country_fixed"]["excluded_ambiguous"]},
             "secs": time.time() - t}
        f.write(json.dumps(o) + "\n"); f.flush()
        print(r["pair"], r["layer"], r["variant"], r.get("seed", 0), "repro", o["repro"],
              "KL %.3f->%.3f  mult %.2f->%.2f" % (o["old"]["kl_mean"], o["new"]["kl_mean"], o["old"]["reward_mult"],
                                                   o["new"]["reward_mult"]), flush=True)


if __name__ == "__main__":
    main()
