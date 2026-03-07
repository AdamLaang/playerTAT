from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 shapes must match, got {z1.shape} and {z2.shape}")

    bsz = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
    sim = sim / tau

    diag = torch.eye(2 * bsz, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, float("-inf"))

    targets = torch.arange(2 * bsz, device=z.device)
    targets = (targets + bsz) % (2 * bsz)

    return F.cross_entropy(sim, targets)


def role_aware_info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    is_goalkeeper: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE where negatives are restricted by role:
    - GK anchors only use outfield negatives
    - outfield anchors only use GK negatives
    """
    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 shapes must match, got {z1.shape} and {z2.shape}")
    if is_goalkeeper.ndim != 1 or is_goalkeeper.shape[0] != z1.shape[0]:
        raise ValueError("is_goalkeeper must have shape [B]")

    bsz = z1.size(0)
    total = 2 * bsz
    z = torch.cat([z1, z2], dim=0)
    roles = torch.cat([is_goalkeeper.bool(), is_goalkeeper.bool()], dim=0)

    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1) / tau
    targets = torch.arange(total, device=z.device)
    targets = (targets + bsz) % total

    losses = []
    for i in range(total):
        pos_idx = int(targets[i].item())
        neg_mask = roles != roles[i]
        neg_mask[i] = False
        if neg_mask.sum() == 0:
            continue

        logits = torch.cat([sim[i, pos_idx].unsqueeze(0), sim[i, neg_mask]], dim=0).unsqueeze(0)
        labels = torch.zeros(1, dtype=torch.long, device=z.device)
        losses.append(F.cross_entropy(logits, labels))

    if not losses:
        return z.new_tensor(0.0)
    return torch.stack(losses).mean()


def mixed_role_info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    is_goalkeeper: torch.Tensor,
    tau: float = 0.07,
    cross_role_weight: float = 1.0,
    in_role_weight: float = 1.0,
) -> torch.Tensor:
    """
    Mixed-role InfoNCE:
    - uses BOTH cross-role and in-role negatives
    - can optionally up/down-weight each negative group
    """
    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 shapes must match, got {z1.shape} and {z2.shape}")
    if is_goalkeeper.ndim != 1 or is_goalkeeper.shape[0] != z1.shape[0]:
        raise ValueError("is_goalkeeper must have shape [B]")
    if cross_role_weight <= 0 or in_role_weight <= 0:
        raise ValueError("cross_role_weight and in_role_weight must be > 0")

    bsz = z1.size(0)
    total = 2 * bsz
    z = torch.cat([z1, z2], dim=0)
    roles = torch.cat([is_goalkeeper.bool(), is_goalkeeper.bool()], dim=0)

    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1) / tau
    targets = torch.arange(total, device=z.device)
    targets = (targets + bsz) % total

    log_cross = torch.log(z.new_tensor(float(cross_role_weight)))
    log_in = torch.log(z.new_tensor(float(in_role_weight)))

    losses = []
    all_idx = torch.arange(total, device=z.device)
    for i in range(total):
        pos_idx = int(targets[i].item())
        valid = all_idx != i
        neg_mask = valid.clone()
        neg_mask[pos_idx] = False
        neg_idx = all_idx[neg_mask]
        if neg_idx.numel() == 0:
            continue

        same_role = roles[neg_idx] == roles[i]
        neg_weights_log = torch.where(same_role, log_in, log_cross)

        pos_logit = sim[i, pos_idx].unsqueeze(0)
        neg_logits = sim[i, neg_idx] + neg_weights_log
        logits = torch.cat([pos_logit, neg_logits], dim=0).unsqueeze(0)
        labels = torch.zeros(1, dtype=torch.long, device=z.device)
        losses.append(F.cross_entropy(logits, labels))

    if not losses:
        return z.new_tensor(0.0)
    return torch.stack(losses).mean()
