"""Phase 3: hop separation. Can an edit make the model output the target capital while the
state-belief question (STATE_Q) still gets the ORIGINAL state? Same edit (layer, position rule,
vector) is applied to both prompt types. STATE_Q_HO (different wording, never used in training the
keep penalty) is the overfitting control: hop2_success_ho uses it instead of STATE_Q.

Methods (vectors built from dev cities only; evaluated on held-out source cities):
  md_city    mean-diff at city_last (capital prompts)                  [reference]
  md_final   mean-diff at final token (capital prompts), applied at final token of every prompt
  md_capital mean-diff at the ' capital' token (absent in STATE_Q -> hop-1 trivially untouched)
  perp_city  md_city with the state-belief mean-diff direction (from STATE_Q prompts) projected out
  grad_city  gradient-trained additive vector at city_last (C2, no penalty)
  hop2_city  gradient-trained at city_last with a penalty to KEEP the source state on STATE_Q
  gradn_city / hop2n_city  same two, norm capped at |md_city| (uncapped Adam lands at ~15x |md|,
             i.e. overwrites the residual; see LAB_NOTEBOOK Phase 4 diagnosis)
"""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__) + "/..")
import time

import torch

from edithunt.common import base_args, save, seed_all, valid_table
from edithunt.data import CAPITALS, STATE_Q, STATE_Q_HO, STATES, TEMPLATES
from edithunt.instances import eligible_states, pick_pairs
from edithunt.metrics import fmt, rate
from edithunt.model import Subject
from edithunt.optim import train_additive
from edithunt.runner import Job, run_jobs
from edithunt.vectors import meandiff

ap = base_args(__doc__)
ap.add_argument("--n_pairs", type=int, default=6)
ap.add_argument("--layers", default="4,8,12,16,18,20,21,22,23,24,26")
ap.add_argument("--grad_layers", default="8,16,20")
ap.add_argument("--steps", type=int, default=60)
args = ap.parse_args()
seed_all(args.seed)
S = Subject(args.model)
V = valid_table(args.model)
caps = list(CAPITALS.values())
layers = [int(x) for x in args.layers.split(",")]
glayers = [int(x) for x in args.grad_layers.split(",")]
states = eligible_states(V, "fs1")
pairs = pick_pairs(states, args.n_pairs, args.seed)
t0 = time.time()


def cap_tok(e) -> list[int]:
    """position of the last ' capital' token (exists only in capital prompts)."""
    idx = [i for i in range(e.city_last + 1, len(e.ids)) if "capital" in S.tok.decode([e.ids[i]])]
    return idx[-1:] if idx else []


POS = {"city": lambda e: [e.city_last], "final": lambda e: [e.final], "capital": cap_tok}


def mean_at(cities, tmpl, where, L):
    encs = [S.encode(tmpl, c) for c in cities]
    caps_ = {L: [POS[where](e) for e in encs]}
    _, st = S.score(encs, caps[:1], capture=caps_)
    return st[L][:, 0].mean(0)


rows = []
for pi, (src, tgt) in enumerate(pairs):
    _, sp = meandiff(S, V, src, tgt, [0], args.seed)
    dev_s, dev_t, test = sp["dev_s"], sp["dev_t"], sp["test_s"]
    methods = {}  # (name, L) -> (vector, where)
    for L in layers:
        for where in ("city", "final", "capital"):
            v = mean_at(dev_t, TEMPLATES["fs1"], where, L) - mean_at(dev_s, TEMPLATES["fs1"], where, L)
            methods[(f"md_{where}", L)] = (v, where)
        vs = mean_at(dev_t, STATE_Q, "city", L) - mean_at(dev_s, STATE_Q, "city", L)
        vc = methods[("md_city", L)][0]
        u = vs / vs.norm()
        methods[("perp_city", L)] = (vc - (vc @ u) * u, "city")
    dev_encs = [S.encode(TEMPLATES[t], c) for c in dev_s for t in ("fs1", "fs2") if V[t][c]]
    keep_encs = [S.encode(STATE_Q, c) for c in dev_s]
    for L in glayers:
        methods[("grad_city", L)] = (train_additive(S, L, "city_last", dev_encs, CAPITALS[tgt], steps=args.steps, seed=args.seed), "city")
        methods[("hop2_city", L)] = (train_additive(S, L, "city_last", dev_encs, CAPITALS[tgt], steps=args.steps,
                                                    keep_encs=keep_encs, keep_answers=[src] * len(keep_encs),
                                                    keep_weight=1.0, seed=args.seed), "city")
        cap = float(methods[("md_city", L)][0].norm())  # grad_layers must be a subset of layers
        methods[("gradn_city", L)] = (train_additive(S, L, "city_last", dev_encs, CAPITALS[tgt], steps=args.steps,
                                                     max_norm=cap, seed=args.seed), "city")
        methods[("hop2n_city", L)] = (train_additive(S, L, "city_last", dev_encs, CAPITALS[tgt], steps=args.steps,
                                                     keep_encs=keep_encs, keep_answers=[src] * len(keep_encs),
                                                     keep_weight=1.0, max_norm=cap, seed=args.seed), "city")
        print(f"  trained L{L} [{time.time()-t0:.0f}s]", flush=True)
    cj, sj = [], []
    for (name, L), (v, where) in methods.items():
        for x in test:
            for tn in ("fs1", "ho_fs"):
                if V[tn][x]:
                    e = S.encode(TEMPLATES[tn], x)
                    p = POS[where](e)
                    cj.append(Job(e, [(L, "add", p, v)] if p else [], dict(m=name, L=L, city=x, T=tn, has_pos=bool(p))))
            for q, qt in (("q", STATE_Q), ("ho", STATE_Q_HO)):
                e = S.encode(qt, x)
                p = POS[where](e)
                sj.append(Job(e, [(L, "add", p, v)] if p else [], dict(m=name, L=L, city=x, q=q, has_pos=bool(p))))
    csc = run_jobs(S, cj, caps, exact=[CAPITALS[src], CAPITALS[tgt]], bs=args.bs)
    ssc = run_jobs(S, sj, STATES, bs=args.bs)
    state_pred = {(j.meta["m"], j.meta["L"], j.meta["city"], j.meta["q"]): STATES[s.argmax()] for j, s in zip(sj, ssc)}
    for j, s in zip(cj, csc):
        m = j.meta
        sp_ = state_pred[(m["m"], m["L"], m["city"], "q")]
        sph = state_pred[(m["m"], m["L"], m["city"], "ho")]
        rows.append(dict(m, pair=f"{src}->{tgt}", pred=caps[s.argmax()], flip=caps[s.argmax()] == CAPITALS[tgt],
                         state_pred=sp_, state_kept=sp_ == src, state_to_tgt=sp_ == tgt, state_kept_ho=sph == src,
                         vnorm=float(methods[(m["m"], m["L"])][0].norm())))
    print(f"pair {pi} {src}->{tgt} test n={len(test)} [{time.time()-t0:.0f}s]", flush=True)

agg = {}
for (name, L) in sorted({(r["m"], r["L"]) for r in rows}):
    R = [r for r in rows if r["m"] == name and r["L"] == L]
    Rf = [r for r in R if r["T"] == "fs1"]
    d = {"flip_fs1": rate([r["flip"] for r in Rf]), "flip_ho_fs": rate([r["flip"] for r in R if r["T"] == "ho_fs"]),
         "state_kept": rate([r["state_kept"] for r in Rf]), "state_to_tgt": rate([r["state_to_tgt"] for r in Rf]),
         "hop2_success": rate([r["flip"] and r["state_kept"] for r in Rf]),
         "state_kept_ho": rate([r["state_kept_ho"] for r in Rf]),
         "hop2_success_ho": rate([r["flip"] and r["state_kept_ho"] for r in Rf]),
         "vnorm": sum(r["vnorm"] for r in R) / len(R)}
    agg[f"{name}|L{L}"] = d
    print(f"{name:10s} L{L:2d} flip {fmt(d['flip_fs1'])} ho {d['flip_ho_fs']['rate']:.2f} state_kept {d['state_kept']['rate']:.2f} "
          f"state->tgt {d['state_to_tgt']['rate']:.2f} HOP2 {fmt(d['hop2_success'])} HOP2ho {d['hop2_success_ho']['rate']:.2f} |v| {d['vnorm']:.1f}")
save(args.model, "phase3_hopsep", vars(args) | {"pairs": pairs}, {"agg": agg, "rows": rows})
