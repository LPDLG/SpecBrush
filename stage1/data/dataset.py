"""Dataset and sidecar-aware runtime view for pigment NPZ files."""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PigmentNPZDataset(Dataset):
    def __init__(self, npz_path: str, index_csv: str = "") -> None:
        super().__init__()
        data = np.load(npz_path, allow_pickle=True)
        self.npz_path = npz_path
        self.x0 = data['x0'].astype(np.float32)
        self.mask = data['mask'].astype(np.float32)

        self.raman = data['raman'].astype(np.float32) if 'raman' in data else None
        self.xrd = data['xrd'].astype(np.float32) if 'xrd' in data else None
        self.raman_peaks = data['raman_peaks'].astype(np.float32) if 'raman_peaks' in data else None
        self.xrd_peaks = data['xrd_peaks'].astype(np.float32) if 'xrd_peaks' in data else None

        self.patch_id = data['meta_patch_id'].astype(np.int64) if 'meta_patch_id' in data else None
        self.t = data['meta_t'].astype(np.int64) if 'meta_t' in data else None
        self.t_start = data['meta_t_start'].astype(np.int64) if 'meta_t_start' in data else self.t
        self.t_end = data['meta_t_end'].astype(np.int64) if 'meta_t_end' in data else self.t
        self.seq_len = data['meta_seq_len'].astype(np.int64) if 'meta_seq_len' in data else None

        self.has_raman = data['meta_has_raman'].astype(np.int64) if 'meta_has_raman' in data else None
        self.has_xrd = data['meta_has_xrd'].astype(np.int64) if 'meta_has_xrd' in data else None
        self.exp_id = data['meta_exp_id'].astype(np.int64) if 'meta_exp_id' in data else None
        self.exp_tag = data['meta_exp_tag'].astype(np.int64) if 'meta_exp_tag' in data else None
        self.exp_hum = data['meta_exp_humidity_median'].astype(np.float32) if 'meta_exp_humidity_median' in data else None

        auto_index = index_csv or self._guess_index_csv(npz_path)
        self.sample_index: Optional[List[Dict[str, str]]] = self._load_index_csv(auto_index) if auto_index and os.path.exists(auto_index) else None

    @staticmethod
    def _guess_index_csv(npz_path: str) -> str:
        stem, _ = os.path.splitext(npz_path)
        candidate = f'{stem}_index.csv'
        if os.path.exists(candidate):
            return candidate
        folder = os.path.dirname(npz_path)
        generic = os.path.join(folder, 'sample_index.csv')
        return generic if os.path.exists(generic) else ''

    @staticmethod
    def _load_index_csv(path: str) -> List[Dict[str, str]]:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))

    def __len__(self) -> int:
        return int(self.x0.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {
            'x0': torch.from_numpy(self.x0[idx]),
            'mask': torch.from_numpy(self.mask[idx]),
        }
        if self.raman is not None:
            item['raman'] = torch.from_numpy(self.raman[idx])
        if self.xrd is not None:
            item['xrd'] = torch.from_numpy(self.xrd[idx])
        if self.raman_peaks is not None:
            item['raman_peaks'] = torch.from_numpy(self.raman_peaks[idx])
        if self.xrd_peaks is not None:
            item['xrd_peaks'] = torch.from_numpy(self.xrd_peaks[idx])
        if self.patch_id is not None:
            item['patch_id'] = torch.tensor(int(self.patch_id[idx]), dtype=torch.long)
        if self.t is not None:
            item['t'] = torch.tensor(int(self.t[idx]), dtype=torch.long)
        if self.t_start is not None:
            item['t_start'] = torch.tensor(int(self.t_start[idx]), dtype=torch.long)
        if self.t_end is not None:
            item['t_end'] = torch.tensor(int(self.t_end[idx]), dtype=torch.long)
        if self.seq_len is not None:
            item['seq_len'] = torch.tensor(int(self.seq_len[idx]), dtype=torch.long)
        if self.has_raman is not None:
            item['has_raman'] = torch.tensor(int(self.has_raman[idx]), dtype=torch.long)
        if self.has_xrd is not None:
            item['has_xrd'] = torch.tensor(int(self.has_xrd[idx]), dtype=torch.long)
        if self.exp_id is not None:
            item['exp_id'] = torch.tensor(int(self.exp_id[idx]), dtype=torch.long)
        if self.exp_tag is not None:
            item['exp_tag'] = torch.tensor(int(self.exp_tag[idx]), dtype=torch.long)
        if self.exp_hum is not None:
            item['exp_hum'] = torch.tensor(float(self.exp_hum[idx]), dtype=torch.float32)
        if self.sample_index is not None and idx < len(self.sample_index):
            row = self.sample_index[idx]
            for key in ('side', 'sequence_parent_id', 'spectral_parent_id', 'augmentation_parent_id', 'split_group_id'):
                if key in row and row[key] != '':
                    item[key] = row[key]
        return item
