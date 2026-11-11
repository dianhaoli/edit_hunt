"""H4(i) hard negatives (pre-registered LAB_NOTEBOOK 2026-10-02): Iso_other measured on cities of states that BORDER
the source (target excluded) vs the grader's random pool, for the same edits. Edits: stored gated keep-trained oracle
(s0-2) on the 11 v2-valid instances, plus the v2.1 library's opt_keep100 / ref_s0 submissions on the pilot set if present.
Neighbour items: all extended-list cities of bordering states x (ho_fs, ho_zs), clean margin >= 0.1, unchanged top
capital = kept. Also reports the gate value on those cities (how often a source-keyed gate opens on a neighbour).
Rows -> results/suite/t2v21/h4_neighbors.json
  PYTHONPATH=. .venv/bin/python experiments/h4_neighbors.py"""
import json

import torch

from edithunt.common import ROOT
from edithunt.data import CITIES_EXT, HELDOUT_TEMPLATES, TEMPLATES
from edithunt.env import ravel as R
from edithunt.env.grader import CAPS, build_ivs, parse
from edithunt.env.instance import load_instance
from edithunt.model import Subject

D = ROOT / "results" / "suite" / "t2v21"
NB = {
    "Alabama": "Florida Georgia Mississippi Tennessee", "Arizona": "California Colorado Nevada New_Mexico Utah",
    "Arkansas": "Louisiana Mississippi Missouri Oklahoma Tennessee Texas", "California": "Arizona Nevada Oregon",
    "Colorado": "Arizona Kansas Nebraska New_Mexico Oklahoma Utah Wyoming", "Connecticut": "Massachusetts New_York Rhode_Island",
    "Delaware": "Maryland New_Jersey Pennsylvania", "Florida": "Alabama Georgia",
    "Georgia": "Alabama Florida North_Carolina South_Carolina Tennessee",
    "Idaho": "Montana Nevada Oregon Utah Washington Wyoming", "Illinois": "Indiana Iowa Kentucky Missouri Wisconsin",
    "Indiana": "Illinois Kentucky Michigan Ohio", "Iowa": "Illinois Minnesota Missouri Nebraska South_Dakota Wisconsin",
    "Kansas": "Colorado Missouri Nebraska Oklahoma",
    "Kentucky": "Illinois Indiana Missouri Ohio Tennessee Virginia West_Virginia", "Louisiana": "Arkansas Mississippi Texas",
    "Maine": "New_Hampshire", "Maryland": "Delaware Pennsylvania Virginia West_Virginia",
    "Massachusetts": "Connecticut New_Hampshire New_York Rhode_Island Vermont", "Michigan": "Indiana Ohio Wisconsin",
    "Minnesota": "Iowa North_Dakota South_Dakota Wisconsin", "Mississippi": "Alabama Arkansas Louisiana Tennessee",
    "Missouri": "Arkansas Illinois Iowa Kansas Kentucky Nebraska Oklahoma Tennessee",
    "Montana": "Idaho North_Dakota South_Dakota Wyoming", "Nebraska": "Colorado Iowa Kansas Missouri South_Dakota Wyoming",
    "Nevada": "Arizona California Idaho Oregon Utah", "New Hampshire": "Maine Massachusetts Vermont",
    "New Jersey": "Delaware New_York Pennsylvania", "New Mexico": "Arizona Colorado Oklahoma Texas Utah",
    "New York": "Connecticut Massachusetts New_Jersey Pennsylvania Vermont",
    "North Carolina": "Georgia South_Carolina Tennessee Virginia", "North Dakota": "Minnesota Montana South_Dakota",
    "Ohio": "Indiana Kentucky Michigan Pennsylvania West_Virginia",
    "Oklahoma": "Arkansas Colorado Kansas Missouri New_Mexico Texas", "Oregon": "California Idaho Nevada Washington",
    "Pennsylvania": "Delaware Maryland New_Jersey New_York Ohio West_Virginia", "Rhode Island": "Connecticut Massachusetts",
    "South Carolina": "Georgia North_Carolina", "South Dakota": "Iowa Minnesota Montana Nebraska North_Dakota Wyoming",
    "Tennessee": "Alabama Arkansas Georgia Kentucky Mississippi Missouri North_Carolina Virginia",
    "Texas": "Arkansas Louisiana New_Mexico Oklahoma", "Utah": "Arizona Colorado Idaho Nevada New_Mexico Wyoming",
    "Vermont": "Massachusetts New_Hampshire New_York", "Virginia": "Kentucky Maryland North_Carolina Tennessee West_Virginia",
    "Washington": "Idaho Oregon", "West Virginia": "Kentucky Maryland Ohio Pennsylvania Virginia",
    "Wisconsin": "Illinois Iowa Michigan Minnesota",
}
NB = {k: [x.replace("_", " ") for x in v.split()] for k, v in NB.items()}


def kept_rate(S, items, edits):
    _, o0, o1 = R._grouped(S, items, edits, CAPS, 16, HELDOUT_TEMPLATES)
    keep = (R._top2_margin(o0) >= R.MIN_MARGIN).tolist()
    k = [bool(o1[i].argmax() == o0[i].argmax()) for i in range(len(items)) if keep[i]]
    return (sum(k) / len(k) if k else float("nan")), len(k)


def gate_open(S, inst, edits, cities):
    """Mean gate value on these cities (fs1 encodings) for a gated edit, else None."""
    e = edits[0]
    if len(e) < 4 or e[3] != "gate":
        return None
    k, b, w, _ = e[2]
    es = [S.encode(TEMPLATES["ho_fs"], c) for c in cities]
    h = S.resid(es, [e[0]], "city_last", 16)[e[0]].float()
    return ((h @ k.float() - float(b)) / float(w)).clamp(0, 1).mean().item()


def main():
    S = Subject("Qwen/Qwen2.5-1.5B")
    rows = []
    valid = json.loads((ROOT / "results/suite/t2_ravel/validate_summary.json").read_text())["valid_instances"]
    Ps = {}
    todo = []
    for pair, L in valid:
        s, t = pair.split("->")
        P = Ps.setdefault(pair, R.make_pair(S, s, t))
        inst = R.make_instance(S, P, L, "v2.1")
        stem = f"{s.replace(' ', '_')}_to_{t.replace(' ', '_')}-L{L}"
        for sd in (0, 1, 2):
            e = R.load_edits(ROOT / f"results/suite/t2_ravel/edits_v/{stem}-oracle-s{sd}.pt")
            todo.append((inst, f"oracle_s{sd}", e))
    for p in sorted((ROOT / "results/suite/instances/T2_keepstate_v21").glob("*.private.json")):
        inst = load_instance(p)
        for m in ("opt_keep100", "ref_s0", "opt_keep100_more"):
            q = D / "subs" / f"{inst.id}__{m}.json"
            if q.exists() and json.loads(q.read_text()):
                e, why = parse(S, inst, json.loads(q.read_text()))
                todo.append((inst, m, e))
    for inst, m, e in todo:
        s, t = inst.source, inst.target
        nb_states = [x for x in NB[s] if x != t]
        nb_cities = [c for x in nb_states for c in CITIES_EXT[x]]
        nb_items = [(c, tk) for c in nb_cities for tk in HELDOUT_TEMPLATES]
        pool_items = [(c, tk) for c in inst.extra["iso_pool"] for tk in HELDOUT_TEMPLATES]
        nk, nn = kept_rate(S, nb_items, e)
        pk, pn = kept_rate(S, pool_items, e)
        r = {"id": inst.id, "layer": inst.constraints["layer"], "method": m, "neighbours": nb_states,
             "iso_neighbour": nk, "n_neighbour": nn, "iso_pool": pk, "n_pool": pn,
             "gate_neighbour": gate_open(S, inst, e, nb_cities), "gate_pool": gate_open(S, inst, e, inst.extra["iso_pool"])}
        rows.append(r)
        print(inst.id, m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("id", "neighbours")}, flush=True)
    (D / "h4_neighbors.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
