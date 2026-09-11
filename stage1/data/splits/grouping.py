"""Group-aware split and leakage guard helpers."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def group_split_indices(groups: Sequence[Tuple], seed: int, val_ratio: float, test_ratio: float):
    assert len(groups) > 0
    uniq = list(dict.fromkeys(groups))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = int(round(n * float(test_ratio))) if test_ratio > 0 else 0
    n_val = int(round(n * float(val_ratio))) if val_ratio > 0 else 0
    if test_ratio > 0 and n_test == 0 and n >= 3:
        n_test = 1
    if val_ratio > 0 and n_val == 0 and n >= 3:
        n_val = 1
    n_test = min(n_test, n)
    n_val = min(n_val, n - n_test)
    test_keys = set(uniq[:n_test])
    val_keys = set(uniq[n_test:n_test + n_val])
    train_keys = set(uniq[n_test + n_val:])
    idx = np.arange(len(groups))
    train_idx = idx[[g in train_keys for g in groups]]
    val_idx = idx[[g in val_keys for g in groups]]
    test_idx = idx[[g in test_keys for g in groups]]
    return train_idx, val_idx, test_idx


def random_split_indices(n: int, seed: int, val_ratio: float, test_ratio: float):
    idx = np.arange(n)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    n_test = int(round(n * float(test_ratio))) if test_ratio > 0 else 0
    n_val = int(round(n * float(val_ratio))) if val_ratio > 0 else 0
    n_test = min(n_test, n)
    n_val = min(n_val, n - n_test)
    return idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test]


def detect_leakage(train_groups: Iterable[str], other_groups: Iterable[str]) -> List[str]:
    train = set(train_groups)
    other = set(other_groups)
    return sorted(train.intersection(other))
