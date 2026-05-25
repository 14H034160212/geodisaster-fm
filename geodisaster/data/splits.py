"""Train/val/test split logic.

Implements the cross-domain protocols from the proposal §8:
    - cross_region   : train on some prefectures, test on others
    - cross_event    : train on some events, test on unseen events
    - cross_hazard   : train on one hazard, test on another
    - global_to_japan: train on global benchmarks, test on Japan events
    - temporal       : train on year T, test on year T+k

Plus the few-shot label fraction sub-sampler used by P4.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Split:
    train: list[dict]
    val: list[dict]
    test: list[dict]

    def summary(self) -> dict:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def by_event(
    patches: list[dict],
    train_events: Iterable[str],
    val_events: Iterable[str] | None = None,
    test_events: Iterable[str] | None = None,
) -> Split:
    train_events = set(train_events)
    val_events = set(val_events or [])
    test_events = set(test_events or [])
    tr = [p for p in patches if p["event_id"] in train_events]
    va = [p for p in patches if p["event_id"] in val_events]
    te = [p for p in patches if p["event_id"] in test_events]
    return Split(train=tr, val=va, test=te)


def by_region(
    patches: list[dict], catalog,
    train_substr: str, test_substr: str,
    val_fraction: float = 0.1,
    seed: int = 1234,
) -> Split:
    """Patches inherit region from their event via the catalog."""
    region_by_event = {e.event_id: e.region for e in catalog}
    train_pool: list[dict] = []
    test_pool: list[dict] = []
    for p in patches:
        region = region_by_event.get(p["event_id"], "")
        if train_substr.lower() in region.lower():
            train_pool.append(p)
        elif test_substr.lower() in region.lower():
            test_pool.append(p)
    rng = random.Random(seed)
    rng.shuffle(train_pool)
    n_val = max(1, int(val_fraction * len(train_pool)))
    return Split(
        train=train_pool[n_val:], val=train_pool[:n_val], test=test_pool,
    )


def few_shot_subsample(
    patches: list[dict],
    fraction: float,
    seed: int = 1234,
    stratify_by_pos: bool = True,
) -> list[dict]:
    """Sample ``fraction`` of the patches. With stratify_by_pos we preserve
    the positive/negative class balance — important for sparse hazards.
    """
    if fraction >= 1.0:
        return list(patches)
    rng = random.Random(seed)
    if not stratify_by_pos:
        n = max(1, int(round(fraction * len(patches))))
        return rng.sample(list(patches), n)

    pos = [p for p in patches if p.get("pos_fraction", 0) > 0]
    neg = [p for p in patches if p.get("pos_fraction", 0) == 0]
    n_pos = max(1, int(round(fraction * len(pos))))
    n_neg = max(0, int(round(fraction * len(neg))))
    rng.shuffle(pos); rng.shuffle(neg)
    return pos[:n_pos] + neg[:n_neg]
