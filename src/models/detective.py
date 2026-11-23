"""Counterfactual SkillGuard detective model.

Pipeline per skill text:
    1. TextEncoder -> token-level H
    2. DecompositionHead -> H_sem, H_noise (token-level)
    3. NoiseSpaceAttention -> R_local, suspicious tokens, head anomalies
    4. BehaviorGNN -> R_behavior, S_behavior
    5. SoftMultiAnchor -> S_anchor (global), S_local (local match)
    6. Toxicity score:
         ToxicScore = alpha * S_anchor + beta * S_local + gamma * S_behavior

When sliding_window.enabled is true, only the text/noise/anchor branch is
windowed. The behavior branch still consumes the full-document behavior graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .anchors import SoftMultiAnchor
from .behavior_gnn import BehaviorGNN
from .encoder import DecompositionHead, TextEncoder, masked_mean
from .noise_attention import NoiseSpaceAttention


@dataclass
class DetectiveOutput:
    H: torch.Tensor                # (B, N, hidden)
    H_sem: torch.Tensor            # (B, N, decomp_dim)
    H_noise: torch.Tensor          # (B, N, decomp_dim)
    h_sem: torch.Tensor            # (B, decomp_dim) pooled
    h_noise: torch.Tensor          # (B, decomp_dim) pooled
    attn_mask: torch.Tensor        # (B, N)
    R_local: torch.Tensor          # (B, decomp_dim)
    S_anchor: torch.Tensor         # (B,)
    S_local: torch.Tensor          # (B,)
    S_behavior: torch.Tensor       # (B,)
    R_behavior: torch.Tensor       # (B, gnn_hidden)
    head_anomaly: torch.Tensor     # (B, M)
    top_head_idx: torch.Tensor     # (B, K_h)
    suspicious_tokens: torch.Tensor  # (B, K_h, K_tok)
    toxic_score: torch.Tensor      # (B,) in [0,1]
    window_index: Optional[torch.Tensor] = None  # (B,) selected sliding-window index, if enabled


class ToxicSkillDetective(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        enc_cfg = cfg["encoder"]
        anc_cfg = cfg["anchors"]
        att_cfg = cfg["noise_attention"]
        beh_cfg = cfg["behavior"]
        sco_cfg = cfg["scoring"]

        self.cfg = cfg
        self.encoder = TextEncoder(enc_cfg["model_name"], enc_cfg["freeze_backbone"])
        hidden_dim = self.encoder.hidden_dim
        decomp_dim = enc_cfg["decomp_dim"]
        self.decomp = DecompositionHead(hidden_dim, decomp_dim)

        self.anchors = SoftMultiAnchor(
            num_anchors=anc_cfg["num_anchors"],
            dim=decomp_dim,
            use_soft=anc_cfg["use_soft"],
        )

        self.noise_attn = NoiseSpaceAttention(
            dim=decomp_dim,
            num_heads=att_cfg["num_heads"],
            top_k_heads=att_cfg["top_k_heads"],
            top_k_tokens=att_cfg["top_k_tokens"],
        )

        self.behavior_enabled = beh_cfg.get("enabled", True)
        if self.behavior_enabled:
            self.behavior_gnn = BehaviorGNN(
                vocab_size=len(beh_cfg["action_vocab"]),
                hidden=beh_cfg["gnn_hidden"],
                num_layers=beh_cfg["gnn_layers"],
                inject_dim=decomp_dim,
            )
        else:
            self.behavior_gnn = None

        self.alpha = float(sco_cfg["alpha"])
        self.beta = float(sco_cfg["beta"])
        self.gamma = float(sco_cfg["gamma"])
        self.decomp_dim = decomp_dim

        sw_cfg = cfg.get("sliding_window", cfg.get("model", {}).get("sliding_window", {}))
        self.sliding_enabled = bool(sw_cfg.get("enabled", False))
        self.window_stride = int(sw_cfg.get("stride", max(1, cfg["data"]["max_token_length"] // 2)))
        max_windows = sw_cfg.get("max_windows")
        self.max_windows = int(max_windows) if max_windows is not None else None
        self.window_aggregation = str(sw_cfg.get("aggregation", "max")).lower()
        if self.window_aggregation != "max":
            raise ValueError("sliding_window.aggregation currently supports only 'max'")

    # ---- Public helpers ----
    def encode(self, texts: list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        enc = self.encoder.tokenize(texts, max_length=max_length, device=device)
        H, mask = self.encoder(enc["input_ids"], enc["attention_mask"])
        return H, mask

    def _limited_window_indices(self, sample_map: torch.Tensor, n_texts: int) -> torch.Tensor:
        if self.max_windows is None:
            return torch.arange(sample_map.numel(), device=sample_map.device)

        counts = [0] * n_texts
        keep: list[int] = []
        for idx, sample_idx in enumerate(sample_map.detach().cpu().tolist()):
            if counts[sample_idx] < self.max_windows:
                keep.append(idx)
                counts[sample_idx] += 1
        return torch.tensor(keep, dtype=torch.long, device=sample_map.device)

    def _sliding_encode(self, texts: list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        stride = max(0, min(self.window_stride, max_length - 1))
        enc = self.encoder.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            stride=stride,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
        sample_map = enc.pop("overflow_to_sample_mapping").to(device)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        keep = self._limited_window_indices(sample_map, len(texts))
        input_ids = input_ids.index_select(0, keep)
        attention_mask = attention_mask.index_select(0, keep)
        sample_map = sample_map.index_select(0, keep)

        H, mask = self.encoder(input_ids, attention_mask)
        return H, mask, sample_map

    def _text_branch(self, H: torch.Tensor, attn_mask: torch.Tensor) -> dict[str, torch.Tensor | dict]:
        H_sem, H_noise = self.decomp(H)

        h_sem = masked_mean(H_sem, attn_mask)
        h_noise = masked_mean(H_noise, attn_mask)

        attn_out = self.noise_attn(H_noise, attn_mask)
        R_local = attn_out["R_local"]

        S_anchor = self.anchors.alignment_sim(h_noise)
        S_local = self.anchors.alignment_sim(R_local)
        s_anchor_unit = (S_anchor + 1.0) * 0.5
        s_local_unit = (S_local + 1.0) * 0.5
        text_score = self.alpha * s_anchor_unit + self.beta * s_local_unit

        return {
            "H_sem": H_sem,
            "H_noise": H_noise,
            "h_sem": h_sem,
            "h_noise": h_noise,
            "attn_out": attn_out,
            "R_local": R_local,
            "S_anchor": S_anchor,
            "S_local": S_local,
            "text_score": text_score,
        }

    def _select_best_windows(
        self,
        branch: dict[str, torch.Tensor | dict],
        sample_map: torch.Tensor,
        n_texts: int,
    ) -> torch.Tensor:
        text_score = branch["text_score"]
        assert isinstance(text_score, torch.Tensor)
        doc_text_score = torch.full(
            (n_texts,),
            torch.finfo(text_score.dtype).min,
            dtype=text_score.dtype,
            device=text_score.device,
        )
        best_window = torch.zeros(n_texts, dtype=torch.long, device=text_score.device)
        for window_idx in range(text_score.numel()):
            doc_idx = int(sample_map[window_idx].item())
            if text_score[window_idx] > doc_text_score[doc_idx]:
                doc_text_score[doc_idx] = text_score[window_idx]
                best_window[doc_idx] = window_idx
        return best_window

    def _behavior_branch(
        self,
        R_local: torch.Tensor,
        node_ids: Optional[torch.Tensor],
        adj: Optional[torch.Tensor],
        node_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.behavior_enabled and node_ids is not None:
            beh = self.behavior_gnn(node_ids, adj, node_mask, R_local)
            return beh["S_behavior"], beh["R_behavior"]

        B = R_local.shape[0]
        S_behavior = torch.zeros(B, device=R_local.device)
        R_behavior = torch.zeros(B, 1, device=R_local.device)
        return S_behavior, R_behavior

    def forward(
        self,
        texts: list[str],
        max_length: int,
        node_ids: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> DetectiveOutput:
        if self.sliding_enabled:
            H_windows, mask_windows, sample_map = self._sliding_encode(texts, max_length=max_length)
            branch = self._text_branch(H_windows, mask_windows)
            best_window = self._select_best_windows(branch, sample_map, len(texts))

            attn_out = branch["attn_out"]
            assert isinstance(attn_out, dict)
            H = H_windows.index_select(0, best_window)
            attn_mask = mask_windows.index_select(0, best_window)
            H_sem = branch["H_sem"].index_select(0, best_window)
            H_noise = branch["H_noise"].index_select(0, best_window)
            h_sem = branch["h_sem"].index_select(0, best_window)
            h_noise = branch["h_noise"].index_select(0, best_window)
            R_local = branch["R_local"].index_select(0, best_window)
            S_anchor = branch["S_anchor"].index_select(0, best_window)
            S_local = branch["S_local"].index_select(0, best_window)
            text_score = branch["text_score"].index_select(0, best_window)
            head_anomaly = attn_out["head_anomaly"].index_select(0, best_window)
            top_head_idx = attn_out["top_head_idx"].index_select(0, best_window)
            suspicious_tokens = attn_out["suspicious_tokens"].index_select(0, best_window)

            S_behavior, R_behavior = self._behavior_branch(R_local, node_ids, adj, node_mask)
            toxic_score = (text_score + self.gamma * S_behavior).clamp(0.0, 1.0)

            return DetectiveOutput(
                H=H,
                H_sem=H_sem,
                H_noise=H_noise,
                h_sem=h_sem,
                h_noise=h_noise,
                attn_mask=attn_mask,
                R_local=R_local,
                S_anchor=S_anchor,
                S_local=S_local,
                S_behavior=S_behavior,
                R_behavior=R_behavior,
                head_anomaly=head_anomaly,
                top_head_idx=top_head_idx,
                suspicious_tokens=suspicious_tokens,
                toxic_score=toxic_score,
                window_index=best_window,
            )

        H, attn_mask = self.encode(texts, max_length=max_length)
        branch = self._text_branch(H, attn_mask)
        attn_out = branch["attn_out"]
        assert isinstance(attn_out, dict)
        H_sem = branch["H_sem"]
        H_noise = branch["H_noise"]
        h_sem = branch["h_sem"]
        h_noise = branch["h_noise"]
        R_local = branch["R_local"]
        S_anchor = branch["S_anchor"]
        S_local = branch["S_local"]

        S_behavior, R_behavior = self._behavior_branch(R_local, node_ids, adj, node_mask)

        # Map cosine sims (in [-1, 1]) to [0, 1] before combining.
        s_anchor_unit = (S_anchor + 1.0) * 0.5
        s_local_unit = (S_local + 1.0) * 0.5
        toxic_score = (
            self.alpha * s_anchor_unit
            + self.beta * s_local_unit
            + self.gamma * S_behavior
        ).clamp(0.0, 1.0)

        return DetectiveOutput(
            H=H,
            H_sem=H_sem,
            H_noise=H_noise,
            h_sem=h_sem,
            h_noise=h_noise,
            attn_mask=attn_mask,
            R_local=R_local,
            S_anchor=S_anchor,
            S_local=S_local,
            S_behavior=S_behavior,
            R_behavior=R_behavior,
            head_anomaly=attn_out["head_anomaly"],
            top_head_idx=attn_out["top_head_idx"],
            suspicious_tokens=attn_out["suspicious_tokens"],
            toxic_score=toxic_score,
            window_index=None,
        )
