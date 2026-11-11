"""H2 proxy-break test (pre-registered LAB_NOTEBOOK 2026-10-02): private downstream state readouts.
1. Screen candidate readouts for clean accuracy on Qwen2.5-1.5B (all extended-list cities; keep >= 0.85).
2. On the 11 v2-valid instances, apply the stored edits (gated keep-trained oracle s0-2 = the recipe family; mean-diff
   as the sensitivity control) and measure, over held-out source cities clean-correct on the readout:
   kept = edited answer still the source state's value; moved = edited answer = target's value.
   Region items only where source and target regions differ.
Rows -> results/suite/t2v21/h2_readouts.json
  PYTHONPATH=. .venv/bin/python experiments/h2_readouts.py"""
import json
import re

from edithunt.common import ROOT
from edithunt.data import ABBR, CITIES_EXT, CITY2STATE_EXT, READOUTS, STATES
from edithunt.env import ravel as R
from edithunt.env.grader import build_ivs
from edithunt.model import Subject

D = ROOT / "results" / "suite" / "t2v21"
REGION = {**{s: "Northeast" for s in ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont",
                                      "New Jersey", "New York", "Pennsylvania"]},
          **{s: "Midwest" for s in ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin", "Iowa", "Kansas", "Minnesota",
                                    "Missouri", "Nebraska", "North Dakota", "South Dakota"]},
          **{s: "South" for s in ["Delaware", "Florida", "Georgia", "Maryland", "North Carolina", "South Carolina",
                                  "Virginia", "West Virginia", "Alabama", "Kentucky", "Mississippi", "Tennessee",
                                  "Arkansas", "Louisiana", "Oklahoma", "Texas"]},
          **{s: "West" for s in ["Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah", "Wyoming",
                                 "Alaska", "California", "Hawaii", "Oregon", "Washington"]}}
assert len(REGION) == 50
DEMO = [("Chicago", "Illinois"), ("Miami", "Florida"), ("Seattle", "Washington")]


def fs(fmt):
    return "\n".join(fmt.format(city=c, state=s, region=REGION[s]) for c, s in DEMO) + "\n" + fmt.split("{state}")[0].split("{region}")[0].format(city="{city}").rstrip()


RO = {  # key -> (template, candidates, answer(state))
    "abbr_addr": (READOUTS["abbr_addr"][0], [ABBR[s] for s in STATES], lambda s: ABBR[s]),
    "region": (fs("{city} is in the {region}."), ["Northeast", "Midwest", "South", "West"], lambda s: REGION[s]),
    "univ": (fs("{city} is in the same state as the University of {state}."), STATES, lambda s: s),
    "plates": (fs("In {city}, cars have {state} license plates."), STATES, lambda s: s),
    "senate": (fs("{city} is represented in the US Senate by the senators from {state}."), STATES, lambda s: s),
    "gov": (fs("The governor responsible for {city} is the Governor of {state}."), STATES, lambda s: s),
}


def main():
    S = Subject("Qwen/Qwen2.5-1.5B")
    out = {"templates": {k: v[0] for k, v in RO.items()}, "screen": {}, "edits": []}
    cities = sorted(CITY2STATE_EXT)
    ok = {}
    for k, (tpl, cands, ans) in RO.items():
        es = []
        for c in cities:
            try:
                es.append(S.encode(tpl, c))
            except AssertionError:
                pass
        sc, _ = S.score(es, cands, bs=32)
        good = {e.city: cands[i] == ans(CITY2STATE_EXT[e.city]) for e, i in zip(es, sc.argmax(1).tolist())}
        ok[k] = good
        out["screen"][k] = {"acc": sum(good.values()) / len(good), "n": len(good)}
        print(k, out["screen"][k], flush=True)
    keep_ro = [k for k in RO if out["screen"][k]["acc"] >= 0.85]
    out["kept_readouts"] = keep_ro
    valid = json.loads((ROOT / "results/suite/t2_ravel/validate_summary.json").read_text())["valid_instances"]
    Ps = {}
    for pair, L in valid:
        s, t = pair.split("->")
        P = Ps.setdefault(pair, R.make_pair(S, s, t))
        inst = R.make_instance(S, P, L, "v2.1")
        stem = f"{s.replace(' ', '_')}_to_{t.replace(' ', '_')}-L{L}"
        for m, sd in [("meandiff", 0), ("oracle", 0), ("oracle", 1), ("oracle", 2)]:
            e = [x[:3] if len(x) > 3 and x[3] == "add" else x
                 for x in R.load_edits(ROOT / f"results/suite/t2_ravel/edits_v/{stem}-{m}-s{sd}.pt")]
            row = {"pair": pair, "layer": L, "method": m, "seed": sd}
            for k in keep_ro:
                tpl, cands, ans = RO[k]
                if k == "region" and REGION[s] == REGION[t]:
                    continue
                cs = [c for c in inst.test_cities if ok[k].get(c)]
                if not cs:
                    continue
                es = [S.encode(tpl, c) for c in cs]
                sc, _ = S.score(es, cands, build_ivs(e, es), bs=32)
                pred = [cands[i] for i in sc.argmax(1).tolist()]
                row[k] = {"kept": sum(p == ans(s) for p in pred) / len(pred), "moved": sum(p == ans(t) for p in pred) / len(pred),
                          "n": len(pred)}
            out["edits"].append(row)
            print(pair, L, m, sd, {k: (round(v["kept"], 2), round(v["moved"], 2)) for k, v in row.items() if isinstance(v, dict)}, flush=True)
    (D / "h2_readouts.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
