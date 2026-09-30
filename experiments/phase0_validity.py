"""Phase 0: validity filter. An instance (city, template) is valid iff the
unpatched model's argmax over the 50 capitals (exact full-sequence log-prob)
is the correct capital. Also reports agreement with greedy decoding."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__) + "/..")
import time

from edithunt.common import base_args, save, seed_all
from edithunt.data import CAPITALS, CITY2STATE, TEMPLATES, STATE_Q, STATES, study_states
from edithunt.metrics import fmt, rate
from edithunt.model import Subject

ap = base_args(__doc__)
args = ap.parse_args()
seed_all(args.seed)
S = Subject(args.model)
caps = list(CAPITALS.values())
cities = list(CITY2STATE)
t0 = time.time()
valid, pred, greedy_ok, summary = {}, {}, {}, {}
for tn, t in list(TEMPLATES.items()) + [("state_q", STATE_Q)]:
    encs = [S.encode(t, c) for c in cities]
    cands, gold = (STATES, [CITY2STATE[c] for c in cities]) if tn == "state_q" else (caps, [CAPITALS[CITY2STATE[c]] for c in cities])
    sc, _ = S.score(encs, cands, bs=args.bs)
    p = [cands[i] for i in sc.argmax(1).tolist()]
    ok = [a == g for a, g in zip(p, gold)]
    gr = S.greedy(encs, max_new=5, bs=args.bs)
    gok = [g.strip().startswith(x) for g, x in zip(gr, gold)]
    agree = [a == b for a, b in zip(ok, gok)]
    valid[tn] = dict(zip(cities, ok)); pred[tn] = dict(zip(cities, p)); greedy_ok[tn] = dict(zip(cities, gok))
    study = [c for c in cities if CITY2STATE[c] in study_states()]
    summary[tn] = {"valid_all": rate(ok), "valid_study": rate([valid[tn][c] for c in study]),
                   "greedy_correct": rate(gok), "argmax_greedy_agree": rate(agree)}
    print(f"{tn:8s} valid {fmt(summary[tn]['valid_all'])}  greedy-correct {fmt(summary[tn]['greedy_correct'])}  agree {fmt(summary[tn]['argmax_greedy_agree'])}  [{time.time()-t0:.0f}s]", flush=True)

per_state = {}
for s in STATES:
    cs = [c for c in cities if CITY2STATE[c] == s]
    per_state[s] = {tn: sum(valid[tn][c] for c in cs) for tn in valid} | {"n": len(cs)}
save(args.model, "phase0_validity", vars(args), {"summary": summary, "valid": valid, "pred": pred,
                                                   "greedy_ok": greedy_ok, "per_state": per_state})
