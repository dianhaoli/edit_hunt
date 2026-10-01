"""Task families for the breadth screen (RAVEL, Huang et al. ACL 2024, arXiv 2402.17700, plus our US cities).

A family = entities, attributes (entity -> value), and per attribute a list of templates. Each template carries a
*verbalizer*: value -> surface strings the model may answer with (value score = max exact log-prob over its
surfaces). Templates whose answer is not the attribute value itself (yes/no questions, UTC offsets) get no
verbalizer and are excluded (counted in `skipped`).

Pre-specified choices (logged in LAB_NOTEBOOK before any data was scored):
- open-ish categoricals (country, language, timezone, industry, work location, country of birth): candidate set =
  values held by >= MIN_COUNT entities of the full RAVEL family (multi-valued "A,B" values excluded);
- occupation Gender Bias has no labels in the RAVEL files: labels are the model's own majority answer across
  templates (RAVEL also labels from model outputs); its "accuracy" is agreement with that majority;
- splits: RAVEL entity_to_split / prompt_to_split, with "train" = train and val+test = held-out.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

RAVEL_DIR = Path.home() / "ravel" / "data"   # git clone github.com/explanare/ravel; tar xzf data.tgz
MIN_COUNT = 10
ENT_CAP = {"city": 800, "city_pop": 800, "occupation": 1000}  # entities scored per family in stage 1 (default 400)
TPL_CAP = (10, 8)      # (train, held-out) format-valid templates scored per attribute in stage 1


@dataclass
class Template:
    text: str          # contains exactly one "%s"
    attr: str
    split: str         # "train" | "test"
    verb: dict         # value -> [surface strings] (without the leading space; see lead())

    @property
    def lead(self) -> str:
        return "" if self.text[-1] in " \"'/(\n" else " "


@dataclass
class Family:
    name: str
    attrs: list[str]
    values: dict                  # attr -> sorted list of candidate values
    ent_attr: dict                # entity -> {attr: value or None (out of candidate set)}
    ent_split: dict               # entity -> "train" | "test"
    templates: dict               # attr -> [Template]
    skipped: dict = field(default_factory=dict)   # attr -> n templates without a usable verbalizer
    self_labeled: list = field(default_factory=list)  # attrs whose labels come from the model
    iso_only: list = field(default_factory=list)      # attrs never used as a target (constant value)


def _load(f: str, what: str):
    return json.loads((RAVEL_DIR / f"ravel_{f}_{what}.json").read_text())


def _top_values(E: dict, attr: str, canon=lambda v: v) -> list[str]:
    c = Counter(canon(e.get(attr, "")) for e in E.values())
    return sorted(v for v, n in c.items() if n >= MIN_COUNT and v and "," not in v)


def _same(vals, extra=None):
    """verbalizer: each value answers as itself (+ optional extra surfaces)."""
    return {v: [v] + list((extra or {}).get(v, [])) for v in vals}


def _sample_templates(texts: list[str], attr: str, p2s: dict, mk, rng) -> tuple[list[Template], int]:
    out, skipped = [], 0
    for t in texts:
        if t.count("%s") != 1:
            skipped += 1
            continue
        v = mk(t)
        if v is None:
            skipped += 1
            continue
        out.append(Template(t, attr, "train" if p2s.get(t, "train") == "train" else "test", v))
    tr = [t for t in out if t.split == "train"]
    te = [t for t in out if t.split == "test"]
    rng.shuffle(tr); rng.shuffle(te)
    return tr[:TPL_CAP[0]] + te[:TPL_CAP[1]], skipped


def _ravel_family(name: str, f: str, spec: dict, rng: random.Random, canon: dict | None = None,
                  self_labeled=(), only=None) -> Family:
    E = _load(f, "entity_attributes")
    P = _load(f, "attribute_to_prompts")
    e2s = _load(f, "entity_to_split")
    p2s = _load(f, "prompt_to_split")
    canon = canon or {}
    values, templates, skipped = {}, {}, {}
    for attr, (vals, mk) in spec.items():
        values[attr] = vals
        templates[attr], skipped[attr] = _sample_templates(P[attr], attr, p2s, lambda t, mk=mk: mk(t, vals), rng)
    ents = sorted(E) if only is None else sorted(set(only) & set(E))
    rng.shuffle(ents)
    ents = sorted(ents[:ENT_CAP.get(name, 400)])
    ent_attr = {}
    for e in ents:
        d = {}
        for attr in spec:
            if attr in self_labeled:
                d[attr] = None  # filled from the model in stage 1
                continue
            v = canon.get(attr, lambda x: x)(E[e].get(attr, ""))
            d[attr] = v if v in values[attr] else None
        ent_attr[e] = d
    return Family(name, list(spec), values, ent_attr,
                  {e: ("train" if e2s.get(e, "train") == "train" else "test") for e in ents},
                  templates, skipped, list(self_labeled))


# ------------------------------------------------------------------ verbalizer rules per attribute
def _plain(t, vals):
    return _same(vals)


def _country(t, vals):
    return {v: [v] + ([v[4:]] if v.startswith("the ") else ["the " + v]) +
            (["USA", "America", "the US"] if v == "United States" else []) +
            (["UK", "England", "Britain"] if v == "United Kingdom" else []) for v in vals}


def _timezone(t, vals):
    if t.rstrip().endswith("UTC") or "UTC" in t:
        return None
    return _same(vals)


def _category(t, vals):
    if re.search(r"\b(YES|Yes|yes|NO|No|no)\b", t):  # "X is a type of animal: YES" (answer is not the value)
        return None
    return {"plant": ["plant", "plants"], "animal": ["animal", "animals"],
            "non-living thing": ["non-living thing", "non-living things", "non-living"]}


def _size(t, vals):
    if re.search(r"millimeter|centimeter|kilometer|\"meter\"|: meter", t):
        return {"mm": ["millimeter", "millimeters"], "cm": ["centimeter", "centimeters"],
                "m": ["meter", "meters"], "km": ["kilometer", "kilometers"]}
    return {"mm": ["mm"], "cm": ["cm"], "m": ["m"], "km": ["km"]}


def _texture(t, vals):
    if '"soft": "' in t:
        return {"soft": ["yes"], "hard": ["no"]}
    if '"hard": "' in t:
        return {"soft": ["no"], "hard": ["yes"]}
    return {"soft": ["soft"], "hard": ["hard"]}


def _gender(t, vals):
    return {"male": ["male", "man", "he", "his", "him", "He", "His", "Male"],
            "female": ["female", "woman", "she", "her", "She", "Her", "Female"]}


def _field(t, vals):
    return _same(vals, {"Medicine": ["Physiology", "physiology", "medicine"], "Economics": ["Economic", "economics"],
                        "Physics": ["physics"], "Chemistry": ["chemistry"], "Literature": ["literature"],
                        "Peace": ["peace"]})


def _industry_canon(v: str) -> str:
    return {"health care": "healthcare"}.get(v.strip().lower(), v.strip().lower())


def _industry(t, vals):
    return {v: [v] + (["health care"] if v == "healthcare" else []) for v in vals}


def city(rng) -> Family:
    E = _load("city", "entity_attributes")
    spec = {"Country": (_top_values(E, "Country"), _country),
            "Continent": (_top_values(E, "Continent"), _plain),
            "Language": (_top_values(E, "Language"), _plain),
            "Timezone": (_top_values(E, "Timezone"), _timezone)}
    return _ravel_family("city", "city", spec, rng)


def city_pop(rng) -> Family:
    """Added 2026-10-01 after `city` failed validity on random (often obscure) RAVEL cities, with the user's go-ahead:
    the 800 most populous RAVEL cities (GeoNames cities15000, data/city_pop_top800.json); everything else as `city`."""
    E = _load("city", "entity_attributes")
    top = [e for e, _ in json.loads((Path(__file__).parent / "data" / "city_pop_top800.json").read_text())["top800"]]
    spec = {"Country": (_top_values(E, "Country"), _country),
            "Continent": (_top_values(E, "Continent"), _plain),
            "Language": (_top_values(E, "Language"), _plain),
            "Timezone": (_top_values(E, "Timezone"), _timezone)}
    return _ravel_family("city_pop", "city", spec, rng, only=top)


def physical_object(rng) -> Family:
    cat = lambda v: {"non-living things": "non-living thing"}.get(v, v)
    spec = {"Category": (["animal", "non-living thing", "plant"], _category),
            "Color": (_top_values(_load("physical_object", "entity_attributes"), "Color"), _plain),
            "Size": (["cm", "km", "m", "mm"], _size),
            "Texture": (["hard", "soft"], _texture)}
    return _ravel_family("physical_object", "physical_object", spec, rng, canon={"Category": cat})


def occupation(rng) -> Family:
    E = _load("occupation", "entity_attributes")
    spec = {"Industry": (_top_values(E, "Industry", _industry_canon), _industry),
            "Work Location": (_top_values(E, "Work Location"), _plain),
            "Gender Bias": (["female", "male"], _gender)}
    return _ravel_family("occupation", "occupation", spec, rng, canon={"Industry": _industry_canon},
                         self_labeled=["Gender Bias"])


def nobel(rng) -> Family:
    E = _load("nobel_prize_winner", "entity_attributes")
    spec = {"Field": (["Chemistry", "Economics", "Literature", "Medicine", "Peace", "Physics"], _field),
            "Country of Birth": (_top_values(E, "Country of Birth"), _country),
            "Gender": (["female", "male"], _gender)}
    return _ravel_family("nobel", "nobel_prize_winner", spec, rng)


def verb(rng) -> Family:
    """Past tense / singular: every verb has its own value; candidate set = the values of the sampled verbs."""
    E = _load("verb", "entity_attributes")
    P = _load("verb", "attribute_to_prompts")
    e2s, p2s = _load("verb", "entity_to_split"), _load("verb", "prompt_to_split")
    ents = sorted(E)
    rng.shuffle(ents)
    ents = sorted(ents[:ENT_CAP.get("verb", 400)])
    fam = Family("verb", ["Past Tense", "Singular"], {}, {}, {}, {}, {})
    for attr in fam.attrs:
        vals = sorted({E[e][attr] for e in ents})
        fam.values[attr] = vals
        fam.templates[attr], fam.skipped[attr] = _sample_templates(P[attr], attr, p2s, lambda t: _same(vals), rng)
    fam.ent_attr = {e: {a: E[e][a] for a in fam.attrs} for e in ents}
    fam.ent_split = {e: ("train" if e2s.get(e, "train") == "train" else "test") for e in ents}
    return fam


def us_cities(rng) -> Family:
    """Our US-city dataset (edithunt.data). Templates converted from {city} to %s. Extra state/abbr templates
    (marked NEW) were added so every attribute has >= 2 train and >= 1 held-out template."""
    from ..data import ABBR, CAPITALS, CITIES, COUNTRY_Q, COUNTRY_Q_ZS, READOUTS, STATE_Q, STATE_Q_HO, TEMPLATES
    conv = lambda s: s.replace("{city}", "%s")
    states = sorted(CAPITALS)
    caps = {s: CAPITALS[s] for s in states}
    T = {
        "state": [(STATE_Q, "train"), ("The city of {city} is located in the state of", "train"),  # NEW
                  (STATE_Q_HO, "test"), ("{city} is a city in the US state of", "test")],          # NEW
        "capital": [(TEMPLATES[k], "train") for k in ("fs1", "fs2", "zs1", "zs2")] +
                   [(TEMPLATES[k], "test") for k in ("ho_fs", "ho_zs")],
        "abbr": [(READOUTS["abbr"][0], "train"), ("Austin, TX\nDenver, CO\nBoston, MA\n{city},", "train"),  # NEW
                 (READOUTS["abbr_addr"][0], "test")],
        "country": [(COUNTRY_Q, "train"), (COUNTRY_Q_ZS, "test")],
    }
    verbs = {"state": _same(states), "capital": {s: [caps[s]] for s in states},
             "abbr": {s: [ABBR[s]] for s in states},
             "country": {"United States": ["the United States", "United States", "America", "the USA", "the US"],
                         "Canada": ["Canada"], "Mexico": ["Mexico"], "France": ["France"], "Japan": ["Japan"],
                         "United Kingdom": ["the United Kingdom", "England"]}}
    fam = Family("us_cities", ["state", "capital", "abbr", "country"],
                 {"state": states, "capital": states, "abbr": states, "country": sorted(verbs["country"])},
                 {}, {}, {}, {a: 0 for a in T}, iso_only=["country"])
    # value of every attribute is expressed as the *state* (capital/abbr are functions of it); country is constant
    for a, ts in T.items():
        fam.templates[a] = [Template(conv(t), a, sp, verbs[a]) for t, sp in ts]
    cities = sorted((c, s) for s in states for c in CITIES[s])
    r = random.Random(0)
    for c, s in cities:
        fam.ent_attr[c] = {"state": s, "capital": s, "abbr": s, "country": "United States"}
        fam.ent_split[c] = "train" if r.random() < 0.5 else "test"
    return fam


FAMILIES = {"city": city, "city_pop": city_pop, "physical_object": physical_object, "occupation": occupation, "nobel": nobel,
            "verb": verb, "us_cities": us_cities}


def load_family(name: str, seed: int = 0) -> Family:
    return FAMILIES[name](random.Random(seed))
