"""Phase 2: meaning & specificity of the mean-difference vector (built on fs1, city_last).
For each pair/layer, on held-out source cities: capital flip (fs1, ho_fs), state-belief answer
(STATE_Q, argmax over 50 states), country prompt (top-token unchanged, KL), generic text KL
(vector added at a matched token position), and third-state leakage. Random norm-matched
vector as reference for every specificity number."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__) + "/..")
import time

import torch

from edithunt.common import base_args, save, seed_all, valid_table
from edithunt.data import (CAPITALS, CITIES, CITY2STATE, COUNTRY_Q, GENERIC_SENTENCES, STATE_Q, STATES,
                           TEMPLATES)
from edithunt.instances import eligible_states, pick_pairs
from edithunt.metrics import fmt, kl, rate
from edithunt.model import Subject
from edithunt.runner import Job, run_jobs
from edithunt.vectors import meandiff, random_like

ap = base_args(__doc__)
ap.add_argument("--n_pairs", type=int, default=12)
ap.add_argument("--layers", default="2,4,6,8,10,12,14,16,18,20,22")
args = ap.parse_args()
seed_all(args.seed)
S = Subject(args.model)
V = valid_table(args.model)
caps = list(CAPITALS.values())
layers = [int(x) for x in args.layers.split(",")]
states = eligible_states(V, "fs1")
pairs = pick_pairs(states, args.n_pairs, args.seed)
gen_encs = [S.encode_text(t) for t in GENERIC_SENTENCES]
GEN_POS = 3  # matched position: 4th token of each sentence (a content word in most)
t0 = time.time()

# clean references (no edit) for KL
city_pool = sorted({c for s in states for c in CITIES[s] if V["fs1"][c]})
clean_country = dict(zip(city_pool, S.logp_final([S.encode(COUNTRY_Q, c) for c in city_pool], bs=args.bs)))
clean_gen = S.logp_final(gen_encs, bs=args.bs)
clean_state_sc, _ = S.score([S.encode(STATE_Q, c) for c in city_pool], STATES, bs=args.bs)
clean_state = dict(zip(city_pool, [STATES[i] for i in clean_state_sc.argmax(1).tolist()]))
print(f"clean refs done [{time.time()-t0:.0f}s]", flush=True)

rows = []
for pi, (src, tgt) in enumerate(pairs):
    vecs, sp = meandiff(S, V, src, tgt, layers, args.seed)
    thirds = [s for s in states if s not in (src, tgt)][pi % 4::8][:2]
    third_c = [c for s in thirds for c in CITIES[s] if V["fs1"][c]][:6]
    cap_jobs, st_jobs, co_jobs, gen_jobs = [], [], [], []
    for L in layers:
        for vn, v in (("meandiff", vecs[L]), ("random", random_like(vecs[L], 1000 * pi + L))):
            for x in sp["test_s"] + third_c:
                kind = "test" if x in sp["test_s"] else "third"
                for tn in ("fs1", "ho_fs"):
                    if V[tn][x]:
                        e = S.encode(TEMPLATES[tn], x)
                        cap_jobs.append(Job(e, [(L, "add", [e.city_last], v)], dict(L=L, vec=vn, city=x, kind=kind, T=tn)))
                e = S.encode(STATE_Q, x)
                st_jobs.append(Job(e, [(L, "add", [e.city_last], v)], dict(L=L, vec=vn, city=x, kind=kind)))
                e = S.encode(COUNTRY_Q, x)
                co_jobs.append(Job(e, [(L, "add", [e.city_last], v)], dict(L=L, vec=vn, city=x, kind=kind)))
            for gi, e in enumerate(gen_encs):
                gen_jobs.append(Job(e, [(L, "add", [min(GEN_POS, e.final)], v)], dict(L=L, vec=vn, gi=gi)))
    sc = run_jobs(S, cap_jobs, caps, exact=[CAPITALS[src], CAPITALS[tgt]], bs=args.bs)
    for j, s in zip(cap_jobs, sc):
        own = CAPITALS[CITY2STATE[j.meta["city"]]]
        p = caps[s.argmax()]
        rows.append(dict(j.meta, pair=f"{src}->{tgt}", probe="capital", pred=p, to_target=p == CAPITALS[tgt], kept=p == own))
    sc = run_jobs(S, st_jobs, STATES, bs=args.bs)
    for j, s in zip(st_jobs, sc):
        p = STATES[s.argmax()]
        rows.append(dict(j.meta, pair=f"{src}->{tgt}", probe="state", pred=p, to_target=p == tgt,
                         kept=p == clean_state[j.meta["city"]], clean=clean_state[j.meta["city"]]))
    _, lp = run_jobs(S, co_jobs, caps[:1], bs=args.bs, return_logp=True)
    for j, l in zip(co_jobs, lp):
        c0 = clean_country[j.meta["city"]]
        rows.append(dict(j.meta, pair=f"{src}->{tgt}", probe="country", kl=kl(c0[None], l[None]).item(),
                         kept=bool(c0.argmax() == l.argmax())))
    _, lp = run_jobs(S, gen_jobs, caps[:1], bs=args.bs, return_logp=True)
    for j, l in zip(gen_jobs, lp):
        c0 = clean_gen[j.meta["gi"]]
        rows.append(dict(j.meta, pair=f"{src}->{tgt}", probe="generic", kl=kl(c0[None], l[None]).item(),
                         kept=bool(c0.argmax() == l.argmax())))
    print(f"pair {pi} {src}->{tgt} [{time.time()-t0:.0f}s]", flush=True)

agg = {}
for L in layers:
    for vn in ("meandiff", "random"):
        R = [r for r in rows if r["L"] == L and r["vec"] == vn]
        d = {}
        for T in ("fs1", "ho_fs"):
            d[f"capital_flip_{T}"] = rate([r["to_target"] for r in R if r["probe"] == "capital" and r["kind"] == "test" and r["T"] == T])
        d["third_capital_to_target"] = rate([r["to_target"] for r in R if r["probe"] == "capital" and r["kind"] == "third" and r["T"] == "fs1"])
        d["third_capital_kept"] = rate([r["kept"] for r in R if r["probe"] == "capital" and r["kind"] == "third" and r["T"] == "fs1"])
        d["state_to_target"] = rate([r["to_target"] for r in R if r["probe"] == "state" and r["kind"] == "test"])
        d["state_kept"] = rate([r["kept"] for r in R if r["probe"] == "state" and r["kind"] == "test"])
        d["third_state_to_target"] = rate([r["to_target"] for r in R if r["probe"] == "state" and r["kind"] == "third"])
        for pr in ("country", "generic"):
            rr = [r for r in R if r["probe"] == pr and (pr == "generic" or r["kind"] == "test")]
            d[f"{pr}_top1_kept"] = rate([r["kept"] for r in rr])
            kls = torch.tensor([r["kl"] for r in rr])
            d[f"{pr}_kl_mean"] = kls.mean().item(); d[f"{pr}_kl_median"] = kls.median().item()
        agg[f"{vn}|L{L}"] = d
        print(f"L{L:2d} {vn:8s} capflip {fmt(d['capital_flip_fs1'])} state->tgt {fmt(d['state_to_target'])} "
              f"state kept {d['state_kept']['rate']:.2f} country kept {d['country_top1_kept']['rate']:.2f} "
              f"KLc {d['country_kl_mean']:.3f} KLg {d['generic_kl_mean']:.3f} third->tgt {d['third_capital_to_target']['rate']:.2f}")
save(args.model, "phase2_meaning", vars(args) | {"pairs": pairs}, {"agg": agg, "rows": rows})
