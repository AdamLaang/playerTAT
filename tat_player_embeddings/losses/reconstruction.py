from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError("pred, target, and mask must have the same shape")

    if mask.sum() == 0:
        return pred.new_tensor(0.0)

    loss = F.huber_loss(pred, target, delta=delta, reduction="none")
    masked = loss * mask.float()
    return masked.sum() / mask.float().sum().clamp(min=1.0)
