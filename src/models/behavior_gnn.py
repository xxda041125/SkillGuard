"""Section 6: Behavior graph GNN.

A small GCN-style network operating on per-skill behavior graphs (extracted by
the heuristic behavior_extractor). Node features are an action embedding plus
the local toxic representation R_local injected to all nodes (Section 6.2).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_adj(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization: D^-1/2 (A + I) D^-1/2."""
    n = A.shape[-1]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    A_hat = A + I
    deg = A_hat.sum(dim=-1).clamp_min(1.0)
    d_inv_sqrt = deg.pow(-0.5)
    D_inv = torch.diag_embed(d_inv_sqrt)
    return D_inv @ A_hat @ D_inv


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim)
        self.W_nei = nn.Linear(in_dim, out_dim)

    def forward(self, H: torch.Tensor, A_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # H: (B, n, d_in)  A_norm: (B, n, n)  mask: (B, n)
        agg = A_norm @ H
        H_new = F.gelu(self.W_self(H) + self.W_nei(agg))
        return H_new * mask.unsqueeze(-1).float()


class BehaviorGNN(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, num_layers: int, inject_dim: int) -> None:
        super().__init__()
        self.action_emb = nn.Embedding(vocab_size, hidden)
        self.inject_proj = nn.Linear(inject_dim, hidden)
        self.layers = nn.ModuleList([GCNLayer(hidden, hidden) for _ in range(num_layers)])
        self.risk_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.hidden = hidden

    def forward(
        self,
        node_ids: torch.Tensor,            # (B, n)
        adj: torch.Tensor,                 # (B, n, n) binary directed adjacency
        node_mask: torch.Tensor,           # (B, n)
        R_local: torch.Tensor,             # (B, D)
    ) -> dict[str, torch.Tensor]:
        # Symmetrize for message passing (we still keep direction info via dual edges)
        A = adj + adj.transpose(-1, -2)
        A = (A > 0).float()
        A_norm = _normalize_adj(A)

        H = self.action_emb(node_ids)                                  # (B, n, hidden)
        inj = self.inject_proj(R_local).unsqueeze(1)                   # (B, 1, hidden)
        H = H + inj
        H = H * node_mask.unsqueeze(-1).float()

        for layer in self.layers:
            H = layer(H, A_norm, node_mask)

        # Mean pool over real nodes
        denom = node_mask.sum(dim=1, keepdim=True).clamp_min(1.0).float()
        R_behavior = (H * node_mask.unsqueeze(-1).float()).sum(dim=1) / denom  # (B, hidden)

        S_behavior = torch.sigmoid(self.risk_head(R_behavior).squeeze(-1))      # (B,) in [0,1]
        return {"R_behavior": R_behavior, "S_behavior": S_behavior}
