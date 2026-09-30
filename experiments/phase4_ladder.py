"""Phase 4: difficulty ladder. Intervention classes x constraints, evaluated on held-out source
cities x templates (fs1 = build template, ho_fs / ho_zs = never used to build anything).

Classes
  C0   full residual paste of ONE target-state dev city at city_last (reference; disallowed in env)
  C1   mean-difference (target dev - source dev), city_last or all city tokens
  C1t  C1 averaged over the two training templates (fs1, fs2)
  C2   gradient-trained additive vector, unconstrained norm (naive: lands at ~15x |C1|) / C2kl with
       KL penalty on generic text
  C2n  C2 with norm capped at |C1| (same layer, all dev) / C2nkl capped + KL penalty (careful gradient)
  C3   DAS-style learned rank-k subspace, coords set to mean target-dev coords
  C4   black-box prompting: prepend a context sentence ("{city} is in {target}.") - no internals
  C4v  prompt-derived vector: resid(city | context) - resid(city | no context), no target cities needed
Constraints swept: layer (report per layer; ceilings applied in analysis), position rule,
n_dev in {1,2,all}, target-city examples (C1 needs them; C2/C4v don't).
Specificity: third-state leakage (other states' test cities -> target capital / kept own capital, fs1),
KL at final token on generic sentences (vector at token 3) and on COUNTRY_Q,
state-belief change on STATE_Q."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__) + "/..")
import time

import torch

from edithunt.common import base_args, save, seed_all, valid_table
from edithunt.data import CAPITALS, CITIES, COUNTRY_CANDS, COUNTRY_Q, GENERIC_SENTENCES, STATE_Q, STATES, TEMPLATES
from edithunt.hooks import Intervention
from edithunt.instances import eligible_states, pick_pairs, split
from edithunt.metrics import binary_kl, kl, p_us, rate
from edithunt.model import Subject, pos_rule
from edithunt.optim import train_additive, train_das
from edithunt.runner import Job, run_jobs

ap = base_args(__doc__)
ap.add_argument("--n_pairs", type=int, default=6)
ap.add_argument("--layers", default="2,3,4,5,6,8,12,15,21")
ap.add_argument("--grad_layers", default="3,4,5,8,15")
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--das_steps", type=int, default=100)  # DAS from random init needs ~50-70 steps (checked L4/L8)
ap.add_argument("--eval_templates", default="fs1,ho_fs,ho_zs")
args = ap.parse_args()
seed_all(args.seed)
S = Subject(args.model)
V = valid_table(args.model)
caps = list(CAPITALS.values())
layers = [int(x) for x in args.layers.split(",")]
glayers = [int(x) for x in args.grad_layers.split(",")]
evalT = args.eval_templates.split(",")
states = eligible_states(V, "fs1")
pairs = pick_pairs(states, args.n_pairs, args.seed)
gen = [S.encode_text(t) for t in GENERIC_SENTENCES[:25]]
gen_pos = [[min(3, e.final)] for e in gen]
clean_gen = S.logp_final(gen, bs=args.bs)
t0 = time.time()


def resid_mean(cities, tn, L_list, which="city_last"):
    encs = [S.encode(TEMPLATES[tn] if tn in TEMPLATES else tn, c) for c in cities]
    return {L: v.mean(0) for L, v in S.resid(encs, L_list, which).items()}


rows = []
for pi, (src, tgt) in enumerate(pairs):
    dev_s_all, test_s = split(V, src, args.seed)
    dev_t_all, _ = split(V, tgt, args.seed)
    test_encs = {tn: [S.encode(TEMPLATES[tn], x) for x in test_s if V[tn][x]] for tn in evalT}
    state_encs = [S.encode(STATE_Q, x) for x in test_s]
    country_encs = [S.encode(COUNTRY_Q, x) for x in test_s]
    thirds = [s for s in states if s not in (src, tgt)][pi % 4::8][:2]
    third_c = [c for s in thirds for c in CITIES[s] if V["fs1"][c]][:6]
    third_encs = [S.encode(TEMPLATES["fs1"], c) for c in third_c]
    third_caps = [CAPITALS[next(s for s in thirds if c in CITIES[s])] for c in third_c]
    clean_pus = p_us(S.score(country_encs, COUNTRY_CANDS, bs=args.bs, exact=True)[0])
    cands = {}  # (cls, L, ndev, pos) -> ("add", v) | ("set", v) | ("swap", (U, coords))
    nall = len(dev_s_all)
    for nd in sorted({1, 2, nall}):
        ds, dt = dev_s_all[:nd], dev_t_all[:nd]
        ndl = "all" if nd == nall else nd  # label: pooled across pairs with different dev sizes
        ms, mt = resid_mean(ds, "fs1", layers), resid_mean(dt, "fs1", layers)
        ms2 = resid_mean([c for c in ds if V["fs2"][c]] or ds, "fs2", layers)
        mt2 = resid_mean([c for c in dt if V["fs2"][c]] or dt, "fs2", layers)
        for L in layers:
            v = mt[L] - ms[L]
            cands[("C1", L, ndl, "city_last")] = ("add", v)
            cands[("C1", L, ndl, "city_all")] = ("add", v)
            cands[("C1t", L, ndl, "city_last")] = ("add", (v + mt2[L] - ms2[L]) / 2)
    # C0: paste of one target dev city residual (per layer)
    r0 = S.resid([S.encode(TEMPLATES["fs1"], dev_t_all[0])], layers)
    for L in layers:
        cands[("C0", L, 1, "city_last")] = ("set", r0[L][0])
    # C4v: prompt-derived vector from dev source cities (no target cities needed)
    # context sentence has the city literally; encode() locates the template's {city} occurrence
    ctx_encs = [S.encode(f"Note that {c} is located in {tgt}. " + TEMPLATES["fs1"], c) for c in dev_s_all]
    rc = {L: v.mean(0) for L, v in S.resid(ctx_encs, layers).items()}
    r_plain = resid_mean(dev_s_all, "fs1", layers)
    for L in layers:
        cands[("C4v", L, "all", "city_last")] = ("add", rc[L] - r_plain[L])
    # C2 / C3 (gradient) on train templates of all dev source cities
    dev_encs = [S.encode(TEMPLATES[t], c) for c in dev_s_all for t in ("fs1", "fs2") if V[t][c]]
    for L in glayers:
        cands[("C2", L, "all", "city_last")] = ("add", train_additive(S, L, "city_last", dev_encs, CAPITALS[tgt], steps=args.steps, seed=args.seed))
        cands[("C2kl", L, "all", "city_last")] = ("add", train_additive(
            S, L, "city_last", dev_encs, CAPITALS[tgt], steps=args.steps, seed=args.seed,
            kl_encs=gen[:8], kl_pos=gen_pos[:8], kl_weight=1.0))
        cap = float(cands[("C1", L, "all", "city_last")][1].norm())  # grad_layers must be a subset of layers
        cands[("C2n", L, "all", "city_last")] = ("add", train_additive(
            S, L, "city_last", dev_encs, CAPITALS[tgt], steps=args.steps, seed=args.seed, max_norm=cap))
        cands[("C2nkl", L, "all", "city_last")] = ("add", train_additive(
            S, L, "city_last", dev_encs, CAPITALS[tgt], steps=args.steps, seed=args.seed, max_norm=cap,
            kl_encs=gen[:8], kl_pos=gen_pos[:8], kl_weight=1.0))
        tgt_encs = [S.encode(TEMPLATES["fs1"], c) for c in dev_t_all]
        for k in (1, 4):
            cands[(f"C3r{k}", L, "all", "city_last")] = ("swap", train_das(S, L, dev_encs, tgt_encs, CAPITALS[tgt], k=k, steps=args.das_steps, seed=args.seed))
        print(f"  pair {pi} trained L{L} [{time.time()-t0:.0f}s]", flush=True)

    def ivs_for(encs, key, rule=None):
        kind, pl = cands[key]
        rule = rule or key[3]
        pos = pos_rule(encs, rule) if rule != "gen" else gen_pos
        return [Intervention(key[1], pos, kind, pl)]

    for key in cands:
        r = dict(cls=key[0], L=key[1], ndev=key[2], pos=key[3], pair=f"{src}->{tgt}")
        for tn in evalT:
            sc, _ = S.score(test_encs[tn], caps, ivs_for(test_encs[tn], key), bs=args.bs, exact=[CAPITALS[src], CAPITALS[tgt]])
            r[f"flips_{tn}"] = [caps[i] == CAPITALS[tgt] for i in sc.argmax(1).tolist()]
        if third_encs:
            st3, _ = S.score(third_encs, caps, ivs_for(third_encs, key), bs=args.bs)
            top3 = [caps[i] for i in st3.argmax(1).tolist()]
            r["third_to_tgt"] = [t == CAPITALS[tgt] for t in top3]
            r["third_kept"] = [t == c for t, c in zip(top3, third_caps)]
        ss, _ = S.score(state_encs, STATES, ivs_for(state_encs, key), bs=args.bs)
        r["state_to_tgt"] = [STATES[i] == tgt for i in ss.argmax(1).tolist()]
        r["state_kept"] = [STATES[i] == src for i in ss.argmax(1).tolist()]
        pu = p_us(S.score(country_encs, COUNTRY_CANDS, ivs_for(country_encs, key), bs=args.bs, exact=True)[0])
        r["kl_country"] = binary_kl(clean_pus, pu).tolist()  # binary KL on P(US), see data.COUNTRY_Q
        if cands[key][0] != "swap":
            lg = S.logp_final(gen, ivs_for(gen, key, "gen"))
            r["kl_generic"] = kl(clean_gen, lg).tolist()
        else:  # swap on generic text: project-and-set at token 3
            lg = S.logp_final(gen, [Intervention(key[1], gen_pos, "swap", cands[key][1])])
            r["kl_generic"] = kl(clean_gen, lg).tolist()
        kind, pl = cands[key]
        r["norm"] = float(pl.norm()) if kind != "swap" else None
        rows.append(r)
    # C4 black-box prompting (layer-free)
    for tn in evalT:
        encs = [S.encode(f"Note that {x} is located in {tgt}. " + TEMPLATES[tn], x) for x in test_s if V[tn][x]]
        sc, _ = S.score(encs, caps, bs=args.bs)
        rows.append(dict(cls="C4", L=-1, ndev="none", pos="prompt", pair=f"{src}->{tgt}", T=tn,
                         **{f"flips_{tn}": [caps[i] == CAPITALS[tgt] for i in sc.argmax(1).tolist()]}))
    print(f"pair {pi} {src}->{tgt} done, {len(cands)} candidates [{time.time()-t0:.0f}s]", flush=True)

# aggregate: per (cls, L, ndev, pos)
agg = {}
for key in sorted({(r["cls"], r["L"], r["ndev"], r["pos"]) for r in rows}, key=str):
    R = [r for r in rows if (r["cls"], r["L"], r["ndev"], r["pos"]) == key]
    d = {}
    for tn in evalT:
        d[f"flip_{tn}"] = rate([f for r in R for f in r.get(f"flips_{tn}", [])])
    if key[0] != "C4":
        d["state_to_tgt"] = rate([f for r in R for f in r["state_to_tgt"]])
        d["third_to_tgt"] = rate([f for r in R for f in r.get("third_to_tgt", [])])
        d["third_kept"] = rate([f for r in R for f in r.get("third_kept", [])])
        ns = [r["norm"] for r in R if r["norm"] is not None]
        if ns:
            d["norm"] = sum(ns) / len(ns)
        d["kl_country"] = float(torch.tensor([x for r in R for x in r["kl_country"]]).mean())
        d["kl_generic"] = float(torch.tensor([x for r in R for x in r["kl_generic"]]).mean())
    agg["|".join(map(str, key))] = d
    print("|".join(map(str, key)), " ".join(f"{k}={v['rate']:.2f}(n={v['n']})" if isinstance(v, dict) else f"{k}={v:.3f}" for k, v in d.items()))
save(args.model, "phase4_ladder", vars(args) | {"pairs": pairs}, {"agg": agg, "rows": rows})
