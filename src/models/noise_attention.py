"""Noise-space multi-head attention with anomaly-driven head & token selection.

Implements section 5 of ToxicSkill_v3_plus_soft_multi_anchor.md:

  5.1 Multi-head attention over H_noise (no skip connection -> outputs Z_m)
  5.2 Per-head anomaly score   a_m = max(A_m) - mean_m max(A_m)
  5.3 Top-K toxic head selection
  5.4 Token-level suspicious region per selected head
  5.5 Aggregated local representation R_local
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class NoiseSpaceAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, top_k_heads: int, top_k_tokens: int) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.top_k_heads = min(top_k_heads, num_heads)
        self.top_k_tokens = top_k_tokens

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        H_noise: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            H_noise:   (B, N, D)
            attn_mask: (B, N) — 1 for real tokens, 0 for padding

        Returns dict with:
            R_local:           (B, D)               — Section 5.5
            head_anomaly:      (B, M)               — a_m
            top_head_idx:      (B, K_h)             — selected heads
            suspicious_tokens: (B, K_h, K_tok)      — token positions
            suspicious_mask:   (B, N)               — union of suspicious tokens
        """
        B, N, D = H_noise.shape
        M = self.num_heads
        d_h = self.head_dim

        Q = self.W_q(H_noise).view(B, N, M, d_h).transpose(1, 2)  # (B, M, N, d_h)
        K = self.W_k(H_noise).view(B, N, M, d_h).transpose(1, 2)
        V = self.W_v(H_noise).view(B, N, M, d_h).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_h)  # (B, M, N, N)

        # Mask out padding tokens in the key dimension
        key_mask = attn_mask.unsqueeze(1).unsqueeze(2).bool()           # (B, 1, 1, N)
        scores = scores.masked_fill(~key_mask, float("-inf"))

        A = torch.softmax(scores, dim=-1)                               # (B, M, N, N)
        A = torch.nan_to_num(A, nan=0.0)                                # rows that are fully padded

        Z = torch.matmul(A, V)                                          # (B, M, N, d_h)

        # ---- Section 5.2: per-head anomaly score
        # s_m = max_{i,j} A[b, m, i, j] (peak attention energy)
        # Restrict to non-padded queries
        query_mask = attn_mask.unsqueeze(1).unsqueeze(-1).bool()        # (B, 1, N, 1)
        A_for_max = A.masked_fill(~query_mask, 0.0)
        s = A_for_max.flatten(2).max(dim=-1).values                     # (B, M)
        mu = s.mean(dim=-1, keepdim=True)                               # (B, 1)
        a = s - mu                                                      # (B, M)

        # ---- Section 5.3: Top-K toxic head selection
        K_h = self.top_k_heads
        top_vals, top_idx = a.topk(K_h, dim=-1)                         # (B, K_h)

        # Gather (Z, A) for top heads
        idx_for_Z = top_idx.view(B, K_h, 1, 1).expand(-1, -1, N, d_h)
        Z_top = torch.gather(Z, dim=1, index=idx_for_Z)                 # (B, K_h, N, d_h)

        idx_for_A = top_idx.view(B, K_h, 1, 1).expand(-1, -1, N, N)
        A_top = torch.gather(A, dim=1, index=idx_for_A)                 # (B, K_h, N, N)

        # ---- Section 5.4: per-head suspicious tokens
        # token importance per head = incoming attention mass received by each key token
        query_mask_top = attn_mask.unsqueeze(1).unsqueeze(-1).bool()     # (B, 1, N_query, 1)
        token_score = A_top.masked_fill(~query_mask_top, 0.0)
        token_importance = token_score.sum(dim=-2)                      # (B, K_h, N_key)
        token_importance = token_importance.masked_fill(~attn_mask.unsqueeze(1).bool(), 0.0)

        K_tok = min(self.top_k_tokens, N)
        sus_vals, sus_tokens = token_importance.topk(K_tok, dim=-1)     # (B, K_h, K_tok)

        # ---- Section 5.5: Local toxic representation aggregation
        # Pool each top head over real tokens then sum across heads
        mask_f = attn_mask.unsqueeze(1).unsqueeze(-1).float()            # (B, 1, N, 1)
        Z_top_pooled = (Z_top * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp_min(1.0)  # (B, K_h, d_h)
        # Project each head's d_h back into D using a per-head linear via flatten + out_proj.
        # Trick: place each pooled head into its slot of a (B, M, d_h) tensor (zeros elsewhere)
        # then concat across the M dimension to (B, M*d_h) and pass through out_proj.
        slot = torch.zeros(B, M, d_h, device=Z.device, dtype=Z.dtype)
        slot.scatter_(1, top_idx.unsqueeze(-1).expand(-1, -1, d_h), Z_top_pooled)
        R_local = self.out_proj(slot.reshape(B, M * d_h))                # (B, D)

        # Aggregate suspicious mask over batch
        sus_mask = torch.zeros(B, N, device=Z.device, dtype=torch.bool)
        sus_mask.scatter_(1, sus_tokens.reshape(B, -1), True)
        sus_mask &= attn_mask.bool()

        return {
            "R_local": R_local,
            "head_anomaly": a,
            "top_head_idx": top_idx,
            "suspicious_tokens": sus_tokens,
            "suspicious_mask": sus_mask,
        }
