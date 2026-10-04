"""Why does the gated oracle miss held-out source cities? (4 TUNING pairs only; not the validation pairs.)
Per held-out (city, wording) item: gate value g = clamp((k.h - b)/w, 0, 1) at the edit site, and whether it flipped.
Also per item: the target's margin under the edit if the gate were forced fully open (g=1)."""
import json
import torch
from edithunt.common import ROOT
from edithunt.data import TEMPLATES
from edithunt.hooks import Intervention
from edithunt.model import Subject
from edithunt.env import ravel as R
from edithunt.env.grader import CAPS

S = Subject("Qwen/Qwen2.5-1.5B")
E = ROOT / "results" / "suite" / "t2_ravel" / "edits"
out = []
for pr in R.candidate_pairs()[:4]:
    P = R.make_pair(S, *pr)
    for L in (8, 12, 16, 20):
        inst = R.make_instance(S, P, L)
        f = E / f"oracle_search-{pr[0].replace(' ', '_')}_to_{pr[1].replace(' ', '_')}-L{L}-gated-s0.pt"
        (L_, pos, (k, b, w, v), kind), = R.load_edits(f)
        ti = CAPS.index(inst.target_capital)
        for c, tk in inst.test_items:
            e = S.encode(TEMPLATES[tk], c)
            h = S.resid([e], [L], "city_last")[L][0].float()
            g = ((h @ k - b) / w).clamp(0, 1).item()
            sc_e = R._scores(S, [e], CAPS, [Intervention(L, [[e.city_last]], "gate", (k, b, w, v))])[0]
            sc_1 = R._scores(S, [e], CAPS, [Intervention(L, [[e.city_last]], "add", v)])[0]  # gate forced open
            out.append({"pair": f"{pr[0]}->{pr[1]}", "layer": L, "city": c, "tpl": tk, "gate": g,
                        "flip": int(sc_e.argmax() == ti), "flip_if_open": int(sc_1.argmax() == ti)})
        x = [o for o in out if o["pair"] == f"{pr[0]}->{pr[1]}" and o["layer"] == L]
        op = [o for o in x if o["gate"] >= 0.99]
        print(pr, L, "items", len(x), "Cause %.2f" % (sum(o["flip"] for o in x) / len(x)),
              "| gate fully open %d/%d" % (len(op), len(x)),
              "| Cause if gate forced open %.2f" % (sum(o["flip_if_open"] for o in x) / len(x)),
              "| Cause among fully-open %.2f" % (sum(o["flip"] for o in op) / max(1, len(op))), flush=True)
(ROOT / "results" / "suite" / "t2_ravel" / "cause_diag.json").write_text(json.dumps(out, indent=0))
