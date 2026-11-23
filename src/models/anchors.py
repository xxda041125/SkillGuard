"""Soft Multi-Anchor module.

Maintains K learnable anchor vectors T = {T_1, ..., T_K} living in the noise
space. Provides:
- max_sim:   max_k sim(h, T_k)
- soft_anchor: T_soft = sum_k softmax(sim(h, T_k))_k * T_k
- soft_sim:  sim(h, T_soft)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMultiAnchor(nn.Module):
    def __init__(self, num_anchors: int, dim: int, use_soft: bool = True) -> None:
        super().__init__()
        self.use_soft = use_soft
        self.anchors = nn.Parameter(torch.randn(num_anchors, dim) * 0.02)

    @property
    def K(self) -> int:
        return self.anchors.shape[0]

    def normalized_anchors(self) -> torch.Tensor:
        return F.normalize(self.anchors, dim=-1)

    def similarity(self, h: torch.Tensor) -> torch.Tensor:
        """Cosine sim between h (B, D) and each anchor -> (B, K)."""
        h_n = F.normalize(h, dim=-1)
        T_n = self.normalized_anchors()
        return h_n @ T_n.t()

    def soft_anchor(self, h: torch.Tensor) -> torch.Tensor:
        sims = self.similarity(h)              # (B, K)
        weights = F.softmax(sims, dim=-1)      # (B, K)
        return weights @ self.anchors          # (B, D)

    def alignment_sim(self, h: torch.Tensor) -> torch.Tensor:
        """Returns sim score in [-1,1] (B,) used in L_align."""
        if self.use_soft:
            T_soft = self.soft_anchor(h)
            return F.cosine_similarity(h, T_soft, dim=-1)
        sims = self.similarity(h)
        return sims.max(dim=-1).values
