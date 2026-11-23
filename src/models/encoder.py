"""Text encoder + semantic / noise decomposition.

Wraps a HuggingFace transformer that produces token-level hidden states H, then
two parallel projection heads decompose H into:
    H_sem   (semantic representation, attack-invariant)
    H_noise (perturbation/attack representation)

Both projections are token-level (R^{n x d}) so that downstream modules
(noise-space attention, suspicious-token localization) can exploit per-token
information.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def _hf_offline_mode() -> bool:
    return bool(
        os.environ.get("TRANSFORMERS_OFFLINE")
        or os.environ.get("HF_HUB_OFFLINE")
    )


class TextEncoder(nn.Module):
    """Transformer backbone -> (token_hidden, attention_mask)."""

    def __init__(self, model_name: str, freeze_backbone: bool = False) -> None:
        super().__init__()
        local_files_only = _hf_offline_mode()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.hidden_dim = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def tokenize(self, texts: list[str], max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state, attention_mask  # (B, N, H), (B, N)


class DecompositionHead(nn.Module):
    """Projects H into H_sem and H_noise via two MLPs."""

    def __init__(self, hidden_dim: int, decomp_dim: int) -> None:
        super().__init__()
        self.sem_proj = nn.Sequential(
            nn.Linear(hidden_dim, decomp_dim),
            nn.GELU(),
            nn.Linear(decomp_dim, decomp_dim),
        )
        self.noise_proj = nn.Sequential(
            nn.Linear(hidden_dim, decomp_dim),
            nn.GELU(),
            nn.Linear(decomp_dim, decomp_dim),
        )

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sem_proj(H), self.noise_proj(H)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool token-level features (B, N, D) with attention mask (B, N) into (B, D)."""
    mask_f = mask.unsqueeze(-1).float()
    summed = (x * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return summed / denom
