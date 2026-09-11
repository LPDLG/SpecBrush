"""Sample index sidecar helpers."""
from __future__ import annotations

import csv
from typing import Dict, Iterable, List


INDEX_COLUMNS = [
    'sample_id',
    'npz_row',
    'source_log',
    'exp_id',
    'exp_tag',
    'side',
    'patch_id',
    'time_index',
    'sequence_parent_id',
    'spectral_parent_id',
    'augmentation_parent_id',
    'is_augmented',
    'split_group_id',
    'has_raman',
    'has_xrd',
]


def write_sample_index(path: str, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in INDEX_COLUMNS})


def load_sample_index(path: str) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def default_split_group(row: Dict[str, object]) -> str:
    if row.get('split_group_id'):
        return str(row['split_group_id'])
    if row.get('spectral_parent_id'):
        return str(row['spectral_parent_id'])
    return f"{row.get('exp_id', '')}:{row.get('patch_id', '')}"
