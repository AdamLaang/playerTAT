from __future__ import annotations

from typing import Dict, List

import torch


def collate_windows(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    keys = batch[0].keys()
    for key in keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out
