"""Phase 7 (task T3 prerequisite): does the subject know other state-level facts about a city?
For each READOUTS key, clean argmax over the 50 answers is correct? Reported over all cities and over
cities valid on the capital template fs1. Readouts with >=85% validity become T3 hidden readouts."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__) + "/..")
from edithunt.common import base_args, save, valid_table
from edithunt.data import CITIES, CITY2STATE, READOUTS, STATES
from edithunt.metrics import rate
from edithunt.model import Subject

ap = base_args(__doc__)
args = ap.parse_args()
S = Subject(args.model)
V = valid_table(args.model)
cities = [c for s in STATES for c in CITIES[s]]
out, valid = {}, {}
for k, (tpl, ans) in READOUTS.items():
    cands = [ans[s] for s in STATES]
    encs = [S.encode(tpl, c) for c in cities]
    sc, _ = S.score(encs, cands, bs=args.bs)
    pred = [STATES[i] for i in sc.argmax(1).tolist()]
    ok = {c: p == CITY2STATE[c] for c, p in zip(cities, pred)}
    valid[k] = ok
    out[k] = {"all": rate(ok.values()), "fs1_valid": rate([ok[c] for c in cities if V["fs1"].get(c)])}
    wrong = [(c, p) for c, p in zip(cities, pred) if not ok[c]][:12]
    print(f"{k:10s} all {out[k]['all']['rate']:.2f} (n={out[k]['all']['n']})  fs1-valid "
          f"{out[k]['fs1_valid']['rate']:.2f} (n={out[k]['fs1_valid']['n']})  wrong e.g. {wrong}", flush=True)
save(args.model, "phase7_readouts", vars(args), {"summary": out, "valid": valid})
