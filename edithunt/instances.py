"""Pair selection and dev/test city splits from the validity table."""
from __future__ import annotations

import random

from .data import CITIES, study_states


def eligible_states(valid: dict, template: str = "fs1", min_valid: int = 5) -> list[str]:
    return [s for s in study_states(min_cities=5)
            if sum(valid[template][c] for c in CITIES[s]) >= min_valid]


def split(valid: dict, state: str, seed: int, template: str = "fs1", n_dev: int | None = None):
    cs = [c for c in CITIES[state] if valid[template][c]]
    rng = random.Random(f"{seed}-{state}")
    rng.shuffle(cs)
    k = n_dev if n_dev is not None else len(cs) // 2
    return cs[:k], cs[k:]


def pick_pairs(states: list[str], n: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pairs = []
    if "Texas" in states and "California" in states:
        pairs.append(("Texas", "California"))
    while len(pairs) < n:
        a, b = rng.sample(states, 2)
        if (a, b) not in pairs:
            pairs.append((a, b))
    return pairs
