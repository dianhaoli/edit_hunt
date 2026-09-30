"""Phase 4b: a leak-aware "careful" method for the difficulty ladder.

Phase 4 found no class with high flip AND low third-state leakage: gradient vectors (C2, C2n) learn an
answer direction (other states' cities -> target capital 0.46-1.00), DAS does not leak but flips 0.47-0.67,
mean-diff sits in between. Here:
  C1      mean-diff (all dev), reference
  C2n     norm-capped gradient (cap = |C1| at that layer), no penalty
  C2nk    C2n + keep term: dev cities of K *other* states must keep their own capital (the agent can do this:
          it only needs cities it is allowed to query)
  C3r1    DAS rank-1, reference
Leakage is evaluated on third states disjoint from the keep states (so C2nk must generalise, not memorise).
Same pairs/splits as Phase 4 (seed 0)."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__) + "/..")
import json
import random
import time

import torch

from edithunt.common import base_args, rdir, save, seed_all, valid_table
from edithunt.data import CAPITALS, CITIES, CITY2STATE, COUNTRY_CANDS, COUNTRY_Q, GENERIC_SENTENCES, STATE_Q, STATES, TEMPLATES
from edithunt.hooks import Intervention
from edithunt.instances import eligible_states, pick_pairs, split
from edithunt.metrics import binary_kl, kl, p_us, rate
from edithunt.model import Subject, pos_rule
from edithunt.optim import train_additive, train_das

ap = base_args(__doc__)
ap.add_argument("--n_pairs", type=int, default=12)
ap.add_argument("--layers", default="4,5,8,15")
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--das_steps", type=int, default=100)
ap.add_argument("--n_keep_states", type=int, default=4)
ap.add_argument("--keep_weight", type=float, default=1.0)
args = ap.parse_args()
seed_all(args.seed)
S = Subject(args.model)
V = valid_table(args.model)
caps = list(CAPITALS.values())
layers = [int(x) for x in args.layers.split(",")]
states = eligible_states(V, "fs1")
pairs = pick_pairs(states, args.n_pairs, args.seed)
gen = [S.encode_text(t) for t in GENERIC_SENTENCES[:25]]
gen_pos = [[min(3, e.final)] for e in gen]
clean_gen = S.logp_final(gen, bs=args.bs)
evalT = ["fs1", "ho_fs"]
t0 = time.time()

ckpt = rdir(args.model) / "phase4b_careful.partial.json"
cfg = vars(args) | {"pairs": pairs}
rows, done = [], set()
if ckpt.exists():
    c = json.loads(ckpt.read_text())
    if c["config"] == json.loads(json.dumps(cfg)):
        rows, done = c["rows"], set(c["done"])
        print(f"resuming: {len(done)} pairs done", flush=True)

for pi, (src, tgt) in enumerate(pairs):
    if pi in done:
        continue
    dev_s, test_s = split(V, src, args.seed)
    dev_t, _ = split(V, tgt, args.seed)
    thirds = [s for s in states if s not in (src, tgt)][pi % 4::8][:2]  # same eval third states as Phase 4
    rng = random.Random(f"{args.seed}-{pi}-keep")
    keep_states = rng.sample([s for s in states if s not in (src, tgt, *thirds)], args.n_keep_states)
    keep_c = [c for s in keep_states for c in split(V, s, args.seed)[0][:2]]
    keep_ct = [(c, t) for c in keep_c for t in ("fs1", "fs2") if V[t][c]]
    keep_encs = [S.encode(TEMPLATES[t], c) for c, t in keep_ct]
    keep_ans = [CAPITALS[CITY2STATE[c]] for c, _ in keep_ct]
    third_c = [c for s in thirds for c in CITIES[s] if V["fs1"][c]][:6]
    test_encs = {tn: [S.encode(TEMPLATES[tn], x) for x in test_s if V[tn][x]] for tn in evalT}
    third_encs = {tn: [(c, S.encode(TEMPLATES[tn], c)) for c in third_c if V[tn][c]] for tn in evalT}
    state_encs = [S.encode(STATE_Q, x) for x in test_s]
    country_encs = [S.encode(COUNTRY_Q, x) for x in test_s]
    clean_pus = p_us(S.score(country_encs, COUNTRY_CANDS, bs=args.bs, exact=True)[0])
    dev_encs = [S.encode(TEMPLATES[t], c) for c in dev_s for t in ("fs1", "fs2") if V[t][c]]
    rs = S.resid([S.encode(TEMPLATES["fs1"], c) for c in dev_s], layers)
    rt = S.resid([S.encode(TEMPLATES["fs1"], c) for c in dev_t], layers)
    tgt_encs = [S.encode(TEMPLATES["fs1"], c) for c in dev_t]
    cands = {}
    for L in layers:
        v1 = rt[L].mean(0) - rs[L].mean(0)
        cap = float(v1.norm())
        cands[("C1", L)] = ("add", v1)
        cands[("C2n", L)] = ("add", train_additive(S, L, "city_last", dev_encs, CAPITALS[tgt], steps=args.steps,
                                                   max_norm=cap, seed=args.seed))
        cands[("C2nk", L)] = ("add", train_additive(S, L, "city_last", dev_encs, CAPITALS[tgt], steps=args.steps,
                                                    max_norm=cap, seed=args.seed, keep_encs=keep_encs,
                                                    keep_answers=keep_ans, keep_weight=args.keep_weight))
        cands[("C3r1", L)] = ("swap", train_das(S, L, dev_encs, tgt_encs, CAPITALS[tgt], k=1, steps=args.das_steps, seed=args.seed))
        print(f"  pair {pi} trained L{L} [{time.time()-t0:.0f}s]", flush=True)

    def ivs(encs, key, pos=None):
        kind, pl = cands[key]
        return [Intervention(key[1], pos if pos is not None else pos_rule(encs, "city_last"), kind, pl)]

    for key, (kind, pl) in cands.items():
        r = dict(cls=key[0], L=key[1], pair=f"{src}->{tgt}", keep_states=keep_states, thirds=thirds,
                 norm=float(pl.norm()) if kind == "add" else None)
        for tn in evalT:
            e = test_encs[tn]
            sc, _ = S.score(e, caps, ivs(e, key), bs=args.bs, exact=[CAPITALS[src], CAPITALS[tgt]])
            r[f"flips_{tn}"] = [caps[i] == CAPITALS[tgt] for i in sc.argmax(1).tolist()]
            te = third_encs[tn]
            if te:
                sc3, _ = S.score([x for _, x in te], caps, ivs([x for _, x in te], key), bs=args.bs)
                top = [caps[i] for i in sc3.argmax(1).tolist()]
                r[f"third_to_tgt_{tn}"] = [t == CAPITALS[tgt] for t in top]
                r[f"third_kept_{tn}"] = [t == CAPITALS[CITY2STATE[c]] for (c, _), t in zip(te, top)]
        ss, _ = S.score(state_encs, STATES, ivs(state_encs, key), bs=args.bs)
        r["state_to_tgt"] = [STATES[i] == tgt for i in ss.argmax(1).tolist()]
        pu = p_us(S.score(country_encs, COUNTRY_CANDS, ivs(country_encs, key), bs=args.bs, exact=True)[0])
        r["kl_country"] = binary_kl(clean_pus, pu).tolist()
        r["kl_generic"] = kl(clean_gen, S.logp_final(gen, ivs(gen, key, gen_pos))).tolist()
        rows.append(r)
    done.add(pi)
    ckpt.write_text(json.dumps({"config": cfg, "done": sorted(done), "rows": rows}, default=float))
    print(f"pair {pi} {src}->{tgt} done [{time.time()-t0:.0f}s]", flush=True)

agg = {}
for key in sorted({(r["cls"], r["L"]) for r in rows}):
    R = [r for r in rows if (r["cls"], r["L"]) == key]
    P = lambda k: rate([f for r in R for f in r.get(k, [])])
    M = lambda k: sum(x for r in R for x in r[k]) / max(1, sum(len(r[k]) for r in R))
    d = {k: P(k) for k in ("flips_fs1", "flips_ho_fs", "third_to_tgt_fs1", "third_kept_fs1", "third_to_tgt_ho_fs",
                           "third_kept_ho_fs", "state_to_tgt")}
    d |= {"kl_country": M("kl_country"), "kl_generic": M("kl_generic")}
    agg[f"{key[0]}|{key[1]}"] = d
    print(f"{key[0]:5s} L{key[1]:2d} flip fs1 {d['flips_fs1']['rate']:.2f} [{d['flips_fs1']['ci95'][0]:.2f},{d['flips_fs1']['ci95'][1]:.2f}] "
          f"(n={d['flips_fs1']['n']}) ho {d['flips_ho_fs']['rate']:.2f} | 3rd->tgt fs1 {d['third_to_tgt_fs1']['rate']:.2f} "
          f"ho {d['third_to_tgt_ho_fs']['rate']:.2f} kept fs1 {d['third_kept_fs1']['rate']:.2f} (n={d['third_kept_fs1']['n']}) "
          f"| state>t {d['state_to_tgt']['rate']:.2f} binKLc {d['kl_country']:.3f} KLg {d['kl_generic']:.3f}")
save(args.model, "phase4b_careful", cfg, {"agg": agg, "rows": rows})
