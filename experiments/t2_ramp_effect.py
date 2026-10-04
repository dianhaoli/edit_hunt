"""Dev-only, effect-based choice of the gate ramp (v trained). LOO over 2 dev cities per instance, 4 TUNING pairs x
L{16,20}. Proxy = cause(held dev city, fs1/zs1) * state(held city, visible STATE_Q) * other(EVAL accessible
other-state cities -- not private, not used in training -- fs1/zs1 top capital unchanged) * (1 - min(1, KL)), KL as in
t2_recipe_select (country on dev cities where P(US) drops + generic on agent-visible sentences).
  PYTHONPATH=. .venv/bin/python experiments/t2_ramp_effect.py --shard 0/2"""
import argparse, copy, json, random, time
from edithunt.common import ROOT
from edithunt.data import CITIES, COUNTRY_CANDS, COUNTRY_Q, STATE_Q, STATES, TEMPLATES, study_states
from edithunt.metrics import binary_kl, kl, p_us
from edithunt.model import Subject
from edithunt.env import ravel as R
from edithunt.env.grader import CAPS, GENERIC_TOOLS, build_ivs

RAMPS = [(0.3, 0.6), (0.2, 0.45), (0.15, 0.35)]
ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); a = ap.parse_args()
K, N = map(int, a.shard.split("/"))
out = ROOT / "results" / "suite" / "t2_ravel" / f"ramp_effect.{K}.jsonl"
done = {(r["pair"], r["layer"], str(tuple(r["ramp"])), r["held"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
f = out.open("a")
S = Subject("Qwen/Qwen2.5-1.5B")
top = lambda e, c, iv=(): R._scores(S, e, c, iv).argmax(1).tolist()
jobs = [(pr, L) for pr in R.candidate_pairs()[:4] for L in (16, 20)]
for j, (pr, L) in enumerate(jobs):
    if j % N != K:
        continue
    P = R.make_pair(S, *pr)
    priv = set(P["iso_pool"]) | set(P["decoys"])
    base = R.make_instance(S, P, L)
    used = {c for cs in base.extra["keep_cities"].values() for c in cs}
    ev = [c for s in study_states(min_cities=5) if s not in pr for c in CITIES[s] if c not in priv and c not in used]
    ev = random.Random(f"{pr}-ev").sample(ev, 60)
    oe = [S.encode(TEMPLATES[t], c) for t in R.DEV_T for c in ev]
    o0 = top(oe, CAPS)
    for held in random.Random(f"{base.id}-loo").sample(base.dev_source, 2):
        inst = copy.deepcopy(base); inst.dev_source = [c for c in base.dev_source if c != held]
        for rp in RAMPS:
            key = (f"{pr[0]}->{pr[1]}", L, str(rp), held)
            if key in done:
                continue
            t = time.time()
            e = R.train_gated(S, inst, L, 0, ramp=rp)
            ce = [S.encode(TEMPLATES[tk], held) for tk in R.DEV_T]
            cause = sum(p == CAPS.index(inst.target_capital) for p in top(ce, CAPS, build_ivs(e, ce))) / len(ce)
            se = [S.encode(STATE_Q, held)]
            state = float(top(se, STATES, build_ivs(e, se))[0] == STATES.index(inst.source))
            other = sum(x == y for x, y in zip(o0, top(oe, CAPS, build_ivs(e, oe)))) / len(oe)
            de = [S.encode(COUNTRY_Q, c) for c in base.dev_source]
            pu0, pu1 = p_us(R._clean_scores(S, de, COUNTRY_CANDS)), p_us(R._scores(S, de, COUNTRY_CANDS, build_ivs(e, de)))
            kc = (binary_kl(pu0, pu1) * (pu1 < pu0).float()).mean().item()
            ge = [S.encode_text(x) for x in GENERIC_TOOLS]
            kg = kl(S.logp_final(ge), S.logp_final(ge, build_ivs(e, ge, [ce[i % len(ce)] for i in range(len(ge))]))).mean().item()
            km = 0.5 * (kc + kg)
            r = {"pair": key[0], "layer": L, "ramp": list(rp), "held": held, "cause": cause, "state": state,
                 "other": other, "kl": km, "proxy": cause * state * other * (1 - min(1, km)), "secs": time.time() - t}
            f.write(json.dumps(r) + "\n"); f.flush()
            print(key, " ".join(f"{x}={r[x]:.2f}" for x in ("cause", "state", "other", "kl", "proxy")), flush=True)
