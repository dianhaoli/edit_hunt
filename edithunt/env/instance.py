"""EditHunt environment instances: generation, tier presets, reference validation, JSON IO.

An instance asks the agent to make `source`-state cities answer the capital of `target` on held-out
cities x held-out templates. The public view (for the agent) omits test cities/templates/items and
the reference result; the private view (for the grader) has everything.

CLI:  python -m edithunt.env.instance --model Qwen/Qwen2.5-0.5B --tier easy --n 3 [--handoff 12]
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from ..common import ROOT, rdir
from ..data import CAPITALS, CITIES, HELDOUT_TEMPLATES, TEMPLATES, study_states
from ..model import Subject
from .grader import CAPS, grade, predict

# Placeholders to tune. max_layer: int, "last", or "handoff[+-k]" (resolved against --handoff).
TIERS = {
    "easy": dict(max_layer="handoff-1", positions=["city_last", "city_all"], max_rank=2, max_norm=None,
                 kl_budget=1.0, fwd_budget=3000, n_dev=4, target_examples=True, preserve_state=False,
                 dev_templates=["fs1", "zs1"]),
    "medium": dict(max_layer=5, positions=["city_last"], max_rank=1, max_norm=None,
                   kl_budget=0.3, fwd_budget=800, n_dev=2, target_examples=True, preserve_state=False,
                   dev_templates=["fs1"]),
    # hop-2 only: capital must flip while the state-belief answer is unchanged
    "hard": dict(max_layer="last", positions=["city_last", "city_all", "final"], max_rank=2, max_norm=None,
                 kl_budget=0.3, fwd_budget=1500, n_dev=3, target_examples=True, preserve_state=True,
                 dev_templates=["fs1", "zs1"]),
}
PUBLIC_KEYS = ("id", "model", "tier", "source", "target", "source_capital", "target_capital",
               "dev_source", "dev_target", "dev_templates", "constraints")


@dataclass
class Instance:
    id: str
    model: str
    tier: str
    source: str
    target: str
    source_capital: str
    target_capital: str
    dev_source: list[str]
    dev_target: list[str]
    dev_templates: dict[str, str]          # key -> template string
    constraints: dict                      # max_layer, positions, max_rank, max_norm, kl_budget, fwd_budget, preserve_state
    # ---- private
    test_cities: list[str] = field(default_factory=list)
    test_templates: dict[str, str] = field(default_factory=dict)
    test_items: list[list[str]] = field(default_factory=list)  # [city, template key], clean-valid only
    ref_pool: dict[str, list[str]] = field(default_factory=dict)  # {"source": [...], "target": [...]}
    reference: dict = field(default_factory=dict)
    leak_items: list[list[str]] = field(default_factory=list)  # [city, template key]: third-state cities (grader leakage)

    def public(self) -> dict:
        return {k: getattr(self, k) for k in PUBLIC_KEYS}

    def private(self) -> dict:
        return asdict(self)

    def heldout_patterns(self) -> list[re.Pattern]:
        """Regexes matching any prompt built from a held-out template."""
        return [re.compile(".*".join(re.escape(p) for p in t.split("{city}")), re.S) for t in self.test_templates.values()]


def resolve_layer(v, n_layers: int, handoff: int) -> int:
    if isinstance(v, int):
        return v
    if v == "last":
        return n_layers - 1
    m = re.fullmatch(r"handoff([+-]\d+)?", v)
    assert m, v
    return max(0, min(n_layers - 1, handoff + int(m.group(1) or 0)))


def validity(S: Subject, cities: list[str], tkeys: list[str], table: dict | None = None, bs: int = 16) -> dict:
    """template -> city -> clean argmax over 50 capitals is correct. Fills `table` in place (computes
    only missing entries)."""
    from ..data import CITY2STATE
    table = table if table is not None else {}
    for tk in tkeys:
        d = table.setdefault(tk, {})
        todo = [c for c in cities if c not in d]
        if todo:
            p = predict(S, [S.encode(TEMPLATES[tk], c) for c in todo], CAPS, (), bs)
            d.update({c: a == CAPITALS[CITY2STATE[c]] for c, a in zip(todo, p)})
    return table


def load_validity(model: str) -> dict:
    p = rdir(model) / "phase0_validity.json"
    return json.loads(p.read_text())["result"]["valid"] if p.exists() else {}


def make_instance(S: Subject, source: str, target: str, tier: str, seed: int, valid: dict,
                  handoff: int, min_test: int = 3, bs: int = 16, max_cities: int | None = None) -> Instance | None:
    """max_cities truncates each state's city list (for tiny smoke tests)."""
    P = TIERS[tier]
    dt = P["dev_templates"]
    C = {s: CITIES[s][:max_cities] for s in (source, target)}
    validity(S, C[source] + C[target], dt + HELDOUT_TEMPLATES, valid, bs)
    b = dt[0]
    src_ok = [c for c in C[source] if valid[b][c]]
    tgt_ok = [c for c in C[target] if valid[b][c]]
    rng = random.Random(f"{seed}-{source}-{target}-{tier}")
    rng.shuffle(src_ok); rng.shuffle(tgt_ok)
    k = P["n_dev"]
    dev_s, test = src_ok[:k], [c for c in C[source] if c not in src_ok[:k]]
    dev_t = tgt_ok[:k] if P["target_examples"] else []
    items = [[c, tk] for c in test for tk in HELDOUT_TEMPLATES if valid[tk][c]]
    others = sorted(s for s in study_states(min_cities=5) if s not in (source, target))
    leak_c = [c for st in rng.sample(others, 2) for c in CITIES[st][:max_cities][:4]]
    validity(S, leak_c, HELDOUT_TEMPLATES, valid, bs)
    leak = [[c, tk] for c in leak_c for tk in HELDOUT_TEMPLATES if valid[tk][c]]
    if len(dev_s) < k or len(items) < min_test or (P["target_examples"] and len(dev_t) < k):
        return None
    cons = {"max_layer": resolve_layer(P["max_layer"], S.n_layers, handoff), "positions": P["positions"],
            "max_rank": P["max_rank"], "max_norm": P["max_norm"], "kl_budget": P["kl_budget"],
            "fwd_budget": P["fwd_budget"], "preserve_state": P["preserve_state"]}
    iid = re.sub(r"\s+", "_", f"{tier}-{source}-{target}-s{seed}")
    return Instance(iid, S.name, tier, source, target, CAPITALS[source], CAPITALS[target], dev_s, dev_t,
                    {t: TEMPLATES[t] for t in dt}, cons, test, {t: TEMPLATES[t] for t in HELDOUT_TEMPLATES},
                    items, {"source": src_ok, "target": tgt_ok}, leak_items=leak)


def reference_vector(S: Subject, inst: Instance, layer: int, bs: int = 16) -> torch.Tensor:
    """Mean over dev templates of mean(target pool) - mean(source pool) at city_last (full private pool)."""
    vs = []
    for t in inst.dev_templates.values():
        m = {}
        for side in ("source", "target"):
            encs = [S.encode(t, c) for c in inst.ref_pool[side]]
            m[side] = S.resid(encs, [layer], "city_last", bs)[layer].mean(0)
        vs.append(m["target"] - m["source"])
    return torch.stack(vs).mean(0)


def validate_instance(S: Subject, inst: Instance, handoff: int, threshold: float = 0.8,
                      layers: list[int] | None = None, bs: int = 16) -> bool:
    """Keep iff the reference intervention reaches held-out flip rate >= threshold. Tries up to 3 layers
    <= min(max_layer, handoff-1); stores the best reference's full grade in inst.reference."""
    top = min(inst.constraints["max_layer"], handoff - 1)
    layers = layers or sorted({top, (2 * top) // 3, top // 3})
    best = None
    for L in layers:
        v = reference_vector(S, inst, L, bs)
        sub = {"edits": [{"layer": L, "position": "city_last", "vector": v.tolist(), "scale": 1.0}]}
        g = grade(S, inst, sub, bs, enforce=False)
        if best is None or g["flip"]["rate"] > best[1]["flip"]["rate"]:
            best = (L, g)
    L, g = best
    inst.reference = {"layer": L, "flip": g["flip"], "grade": g, "threshold": threshold, "layers_tried": layers}
    return g["flip"]["rate"] >= threshold


def generate(S: Subject, tier: str, n: int, seed: int, handoff: int, valid: dict | None = None,
             pairs: list[tuple[str, str]] | None = None, validate: bool = True, max_tries: int = 30,
             bs: int = 16, max_cities: int | None = None) -> list[Instance]:
    valid = valid if valid is not None else load_validity(S.name)
    rng = random.Random(seed)
    states = study_states(min_cities=5)
    out, tries = [], 0
    while len(out) < n and tries < max_tries:
        src, tgt = pairs[tries] if pairs and tries < len(pairs) else rng.sample(states, 2)
        tries += 1
        inst = make_instance(S, src, tgt, tier, seed, valid, handoff, bs=bs, max_cities=max_cities)
        if inst is None:
            print(f"skip {src}->{tgt}: not enough valid cities", flush=True); continue
        ok = validate_instance(S, inst, handoff, bs=bs) if validate else True
        print(f"{inst.id}: ref layer {inst.reference.get('layer')} flip {inst.reference.get('flip', {}).get('rate')} "
              f"-> {'keep' if ok else 'drop'}", flush=True)
        if ok:
            out.append(inst)
    return out


def save_instance(inst: Instance, d: Path) -> tuple[Path, Path]:
    d.mkdir(parents=True, exist_ok=True)
    pub, priv = d / f"{inst.id}.public.json", d / f"{inst.id}.private.json"
    pub.write_text(json.dumps(inst.public(), indent=1))
    priv.write_text(json.dumps(inst.private(), indent=1, default=float))
    return pub, priv


def load_instance(p: str | Path) -> Instance:
    return Instance(**json.loads(Path(p).read_text()))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--tier", default="easy", choices=list(TIERS))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--handoff", type=int, default=None, help="handoff layer (default n_layers//2)")
    ap.add_argument("--pairs", default="", help="e.g. 'Texas:California,Ohio:Oregon'")
    ap.add_argument("--no_validate", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_cities", type=int, default=None, help="truncate city lists (smoke tests)")
    a = ap.parse_args()
    S = Subject(a.model, device=a.device)
    h = a.handoff if a.handoff is not None else S.n_layers // 2
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",") if p]
    d = ROOT / "results" / "instances" / a.model.split("/")[-1] / a.tier
    for inst in generate(S, a.tier, a.n, a.seed, h, pairs=pairs or None, validate=not a.no_validate, bs=a.bs,
                         max_cities=a.max_cities):
        print("saved", *save_instance(inst, d))
