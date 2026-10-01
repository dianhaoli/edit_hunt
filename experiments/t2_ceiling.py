"""Ceiling oracle for the T2 v2 pilot instances: best achievable legal edit with privileged training data,
2-fold cross-fitted so nothing is graded on what it trained on.
Per instance: held-out source cities split in two halves (seeded). For each half A: train configs {learned gate,
fixed gate} on dev + A cities (privileged wordings/readouts) + private pool cities NOT in the grader's Iso sample;
select the config by its grade on A (training data); grade the selected edit on B (cities never trained on) through
the normal submission check (enforce=True). Ceiling = mean over the two halves of the product reward.
  PYTHONPATH=. .venv/bin/python experiments/t2_ceiling.py --shard 0/2"""
import argparse, copy, json, random, time
from edithunt.common import ROOT
from edithunt.model import Subject
from edithunt.env import ravel as R
from edithunt.env.instance import load_instance

D = ROOT / "results" / "suite" / "t2_ravel"
CONFIGS = {"learned_gate": dict(learn_gate=True), "fixed_gate": dict(learn_gate=False)}
ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
K, N = map(int, a.shard.split("/"))
out = D / f"ceiling.{K}.jsonl"
done = {(r["id"], r["fold"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
f = out.open("a")
S = Subject("Qwen/Qwen2.5-1.5B")
paths = sorted((ROOT / "results" / "suite" / "instances" / "T2_keepstate_v2").glob("*.private.json"))


def restrict(inst, cities):
    x = copy.deepcopy(inst)
    x.test_items = [it for it in inst.test_items if it[0] in cities]
    x.test_cities = sorted(cities)
    return x


for j, p in enumerate(paths):
    if j % N != K:
        continue
    inst = load_instance(p)
    L = inst.constraints["layer"]
    cities = sorted(inst.test_cities)
    random.Random(f"{inst.id}-ceil").shuffle(cities)
    halves = [cities[: len(cities) // 2], cities[len(cities) // 2:]]
    iso = set(R.iso_sample(inst, 0))
    pool_train = [c for c in inst.extra["iso_pool"] if c not in iso]
    for fold in (0, 1):
        if (inst.id, fold) in done:
            continue
        A, B = halves[fold], halves[1 - fold]
        fit = list(inst.dev_source) + A
        best = None
        t0 = time.time()
        for name, kw in CONFIGS.items():
            e = R.train_ceiling(S, inst, L, fit, pool_train, seed=fold, **kw)
            gA = R.grade_ravel(S, restrict(inst, set(A)), R.as_submission(e), enforce=True)
            if not gA.get("valid"):
                print(inst.id, fold, name, "INVALID", gA.get("reason"), flush=True); continue
            print(inst.id, fold, name, "train-half R=%.3f" % gA["reward_mult"], flush=True)
            if best is None or gA["reward_mult"] > best[1]:
                best = (name, gA["reward_mult"], e)
        gB = R.grade_ravel(S, restrict(inst, set(B)), R.as_submission(best[2]), enforce=True)
        r = {"id": inst.id, "fold": fold, "config": best[0], "train_R": best[1], "n_fit": len(A), "n_eval": len(B),
             "R": gB["reward_mult"], **{k: gB[k]["rate"] for k in ("Cause", "Iso_state", "Iso_other")},
             "kl_mean": gB["kl_mean"], "secs": time.time() - t0}
        f.write(json.dumps(r) + "\n"); f.flush()
        print(inst.id, "fold", fold, best[0], "held-out half R=%.3f" % r["R"],
              {k: round(r[k], 2) for k in ("Cause", "Iso_state", "Iso_other", "kl_mean")}, flush=True)
