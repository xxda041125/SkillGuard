"""Single-document detector and paired-memory scoring."""
from __future__ import annotations
import threading
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from src.utils import build_behavior_batch, load_config, select_device
from src.data.behavior_extractor import extract_behavior_graph
from src.models.factory import build_model


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _minmax_normalize(value: float, stats: dict[str, float]) -> float:
    lo = float(stats["min"])
    hi = float(stats["max"])
    scaled = (float(value) - lo) / max(hi - lo, 1e-12)
    return max(0.0, min(1.0, scaled))


_RISK_CHAIN_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("information_disclosure", ("read", "encode", "upload")),
    ("information_disclosure", ("read", "encode", "send")),
    ("information_disclosure", ("read", "upload")),
    ("information_disclosure", ("read", "send")),
    ("information_disclosure", ("read", "export")),
    ("remote_code_execution", ("download", "execute")),
    ("remote_code_execution", ("download", "run")),
    ("remote_code_execution", ("fetch", "execute")),
    ("remote_code_execution", ("fetch", "run")),
    ("remote_code_execution", ("decode", "execute")),
    ("remote_code_execution", ("decrypt", "execute")),
    ("privilege_or_persistence", ("write", "source")),
    ("privilege_or_persistence", ("write", "execute")),
    ("supply_chain_execution", ("install", "execute")),
)


def _compact_snippet(text: str, start: int, end: int, context: int = 80, max_chars: int = 260) -> str:
    start = max(0, int(start))
    end = max(start, int(end))
    left = max(0, start - context)
    right = min(len(text), end + context)
    snippet = " ".join(text[left:right].split())
    if len(snippet) <= max_chars:
        return snippet
    return snippet[: max_chars - 3].rstrip() + "..."


def _find_ordered_match(nodes: list[str], pattern: tuple[str, ...], start_at: int = 0) -> list[int] | None:
    indices: list[int] = []
    pos = start_at
    for action in pattern:
        while pos < len(nodes) and nodes[pos] != action:
            pos += 1
        if pos >= len(nodes):
            return None
        indices.append(pos)
        pos += 1
    return indices


def _extract_behavior_chain_evidence(
    text: str,
    vocab: list[str],
    max_nodes: int,
    max_chains: int = 8,
) -> list[dict[str, Any]]:
    graph = extract_behavior_graph(text, vocab=vocab, max_nodes=max_nodes)
    if not graph.nodes or (len(graph.nodes) == 1 and graph.spans == [(0, 0)]):
        return []

    chains: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for risk_type, pattern in _RISK_CHAIN_RULES:
        for start_at in range(len(graph.nodes)):
            matched = _find_ordered_match(graph.nodes, pattern, start_at=start_at)
            if matched is None:
                break
            key = (risk_type, tuple(matched))
            if key in seen:
                continue
            seen.add(key)
            spans = [graph.spans[idx] for idx in matched]
            span_start = min(start for start, _ in spans)
            span_end = max(end for _, end in spans)
            chains.append(
                {
                    "risk_type": risk_type,
                    "actions": list(pattern),
                    "node_indices": matched,
                    "spans": [[int(start), int(end)] for start, end in spans],
                    "snippet": _compact_snippet(text, span_start, span_end),
                }
            )
            if len(chains) >= max_chains:
                return chains
    return chains


class DetectorGate:
    """Thread-safe wrapper around a trained ToxicSkillDetective checkpoint."""

    def __init__(self, checkpoint_path: str | Path, config_path: str | Path,
                 threshold: float = 0.40,
                 include_evidence: bool = False) -> None:
        self.cfg = load_config(config_path)
        self.device = select_device(self.cfg["runtime"]["device"])
        self.threshold = float(threshold)
        self.include_evidence = bool(include_evidence)
        self.sliding_window = dict(
            self.cfg.get("sliding_window")
            or self.cfg.get("model", {}).get("sliding_window", {})
            or {}
        )
        self.sliding_window_enabled = bool(self.sliding_window.get("enabled", False))
        self.model = build_model(self.cfg).to(self.device)
        state = torch.load(str(checkpoint_path), map_location=self.device,
                           weights_only=True)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        self._lock = threading.Lock()

    def _empty_evidence(self) -> dict[str, list[dict[str, Any]]]:
        return {"suspicious_tokens": [], "suspicious_behavior_chains": []}

    def _tokenizer_window(self, skill_md_text: str, max_length: int, window_index: int) -> tuple[list[int], list[Any]]:
        tokenizer = self.model.encoder.tokenizer
        kwargs: dict[str, Any] = {
            "max_length": max_length,
            "padding": True,
            "truncation": True,
            "return_offsets_mapping": True,
        }
        if self.sliding_window_enabled:
            stride = max(0, min(int(self.sliding_window.get("stride", max_length // 2)), max_length - 1))
            kwargs.update({"stride": stride, "return_overflowing_tokens": True})
        enc = tokenizer([skill_md_text], **kwargs)
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        idx = max(0, min(int(window_index), len(input_ids) - 1))
        return list(input_ids[idx]), list(offsets[idx])

    def _selected_window_index(self, out: Any) -> int:
        window_index = getattr(out, "window_index", None)
        if window_index is None:
            return 0
        try:
            return int(window_index[0].detach().cpu().item())
        except Exception:
            return 0

    def _extract_token_evidence(
        self,
        skill_md_text: str,
        out: Any,
        max_length: int,
        max_items: int = 24,
    ) -> list[dict[str, Any]]:
        suspicious_tokens = getattr(out, "suspicious_tokens", None)
        if suspicious_tokens is None:
            return []

        token_positions = suspicious_tokens[0].detach().cpu().tolist()
        top_heads = getattr(out, "top_head_idx", None)
        head_indices = (
            top_heads[0].detach().cpu().tolist()
            if top_heads is not None
            else list(range(len(token_positions)))
        )
        window_index = self._selected_window_index(out)

        try:
            input_ids, offsets = self._tokenizer_window(skill_md_text, max_length, window_index)
        except Exception:
            input_ids, offsets = [], []

        tokenizer = self.model.encoder.tokenizer
        evidence: list[dict[str, Any]] = []
        seen: set[tuple[int, int, int]] = set()
        for head_rank, row in enumerate(token_positions):
            head_idx = int(head_indices[head_rank]) if head_rank < len(head_indices) else head_rank
            for token_rank, token_idx_raw in enumerate(row):
                token_idx = int(token_idx_raw)
                item: dict[str, Any] = {
                    "head": head_idx,
                    "head_rank": head_rank,
                    "token_rank": token_rank,
                    "token_index": token_idx,
                    "window_index": window_index,
                }
                if 0 <= token_idx < len(input_ids):
                    token_id = int(input_ids[token_idx])
                    try:
                        item["token"] = tokenizer.decode([token_id], clean_up_tokenization_spaces=True).strip()
                    except Exception:
                        item["token_id"] = token_id
                    if 0 <= token_idx < len(offsets):
                        offset = offsets[token_idx]
                        start, end = int(offset[0]), int(offset[1])
                        if start == end:
                            continue
                        key = (start, end, head_idx)
                        if key in seen:
                            continue
                        seen.add(key)
                        item.update(
                            {
                                "char_start": start,
                                "char_end": end,
                                "snippet": _compact_snippet(skill_md_text, start, end),
                            }
                        )
                evidence.append(item)
                if len(evidence) >= max_items:
                    return evidence
        return evidence

    def _build_evidence(
        self,
        skill_md_text: str,
        out: Any,
        max_length: int,
        vocab: list[str],
        max_nodes: int,
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            "suspicious_tokens": self._extract_token_evidence(skill_md_text, out, max_length),
            "suspicious_behavior_chains": _extract_behavior_chain_evidence(
                skill_md_text,
                vocab=vocab,
                max_nodes=max_nodes,
            ),
        }

    def _attach_evidence(
        self,
        verdict: dict[str, Any],
        skill_md_text: str,
        out: Any,
        max_length: int,
        vocab: list[str],
        max_nodes: int,
    ) -> dict[str, Any]:
        if self.include_evidence:
            verdict["evidence"] = self._build_evidence(
                skill_md_text,
                out=out,
                max_length=max_length,
                vocab=vocab,
                max_nodes=max_nodes,
            )
        return verdict

    @torch.no_grad()
    def __call__(self, skill_md_text: str, **_: Any) -> dict:
        if not skill_md_text or not skill_md_text.strip():
            verdict = {
                "toxic": False, "score": 0.0,
                "S_anchor": 0.0, "S_local": 0.0, "S_behavior": 0.0,
                "threshold": self.threshold, "skipped": True,
                "sliding_window_enabled": self.sliding_window_enabled,
                "sliding_window": self.sliding_window,
            }
            if self.include_evidence:
                verdict["evidence"] = self._empty_evidence()
            return verdict
        with self._lock:
            max_len = self.cfg["data"]["max_token_length"]
            vocab = self.cfg["behavior"]["action_vocab"]
            max_nodes = self.cfg["behavior"]["max_nodes"]
            beh = build_behavior_batch([skill_md_text], vocab=vocab,
                                        max_nodes=max_nodes, device=self.device)
            out = self.model(
                [skill_md_text],
                max_length=max_len,
                node_ids=beh["node_ids"],
                adj=beh["adj"],
                node_mask=beh["node_mask"],
            )
            score = float(out.toxic_score[0].item())
            verdict = {
                "toxic": bool(score >= self.threshold),
                "score": score,
                "base_score": score,
                "compact3_score": score,
                "detector_score_name": "base_score",
                "S_anchor":   float(out.S_anchor[0].item()),
                "S_local":    float(out.S_local[0].item()),
                "S_behavior": float(out.S_behavior[0].item()),
                "threshold":  self.threshold,
                "sliding_window_enabled": self.sliding_window_enabled,
                "sliding_window": self.sliding_window,
            }
            return self._attach_evidence(
                verdict,
                skill_md_text=skill_md_text,
                out=out,
                max_length=max_len,
                vocab=vocab,
                max_nodes=max_nodes,
            )


class CounterfactualSoftGateDetector(DetectorGate):
    """Online detector using the counterfactual soft-gated score.

    The base encoder sees SKILL.md text. The additional lift
    score compares the current h_noise vector against train-set attacked vs
    paired-clean prototypes, then combines train-minmax-normalized scores as:

        0.75 * base_score_norm + 0.25 * lift_score_norm
    """

    @torch.no_grad()
    def _score_entries(self, entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], torch.Tensor]:
        max_len = self.cfg["data"]["max_token_length"]
        vocab = self.cfg["behavior"]["action_vocab"]
        max_nodes = self.cfg["behavior"]["max_nodes"]
        rows: list[dict[str, Any]] = []
        vectors: list[torch.Tensor] = []
        for start in range(0, len(entries), self.batch_size):
            batch = entries[start : start + self.batch_size]
            texts = [_read_text(Path(row["path"])) for row in batch]
            beh = build_behavior_batch(texts, vocab=vocab, max_nodes=max_nodes, device=self.device)
            out = self.model(
                texts,
                max_length=max_len,
                node_ids=beh["node_ids"],
                adj=beh["adj"],
                node_mask=beh["node_mask"],
            )
            if self.representation == "h_noise":
                z = out.h_noise
            elif self.representation == "r_local":
                z = out.R_local
            else:
                raise ValueError(f"Unknown soft-gate representation: {self.representation}")
            vectors.append(z.detach().cpu())
            base_scores = out.toxic_score.detach().cpu().tolist()
            for idx, row in enumerate(batch):
                rows.append({
                    **row,
                    "base_score": float(base_scores[idx]),
                    "compact3_score": float(base_scores[idx]),
                })
        if not vectors:
            return rows, torch.empty(0)
        return rows, torch.cat(vectors, dim=0)

    def _soft_score(self, base_score: float, vector: torch.Tensor) -> dict[str, float]:
        z = F.normalize(vector.unsqueeze(0), dim=-1)
        lift_score = float(((z @ self.attack_bank.t()) - (z @ self.paired_clean_bank.t())).max().item())
        base_norm = _minmax_normalize(base_score, self.compact_stats)
        lift_norm = _minmax_normalize(lift_score, self.lift_stats)
        soft_score = 0.75 * base_norm + 0.25 * lift_norm
        return {
            "score": float(soft_score),
            "counterfactual_soft_gate_score": float(soft_score),
            "soft_gated_cf_score": float(soft_score),
            "base_score": float(base_score),
            "compact3_score": float(base_score),
            "lift_score": float(lift_score),
            "base_score_norm": float(base_norm),
            "compact3_score_norm": float(base_norm),
            "lift_score_norm": float(lift_norm),
        }

    @torch.no_grad()
    def __call__(self, skill_md_text: str, **_: Any) -> dict:
        if not skill_md_text or not skill_md_text.strip():
            verdict = {
                "toxic": False, "score": 0.0,
                "base_score": 0.0, "compact3_score": 0.0,
                "counterfactual_soft_gate_score": 0.0, "soft_gated_cf_score": 0.0,
                "lift_score": 0.0, "S_anchor": 0.0, "S_local": 0.0, "S_behavior": 0.0,
                "threshold": self.threshold, "skipped": True,
                "detector_score_name": "counterfactual_soft_gate_score",
                "sliding_window_enabled": self.sliding_window_enabled,
                "sliding_window": self.sliding_window,
            }
            if self.include_evidence:
                verdict["evidence"] = self._empty_evidence()
            return verdict
        with self._lock:
            max_len = self.cfg["data"]["max_token_length"]
            vocab = self.cfg["behavior"]["action_vocab"]
            max_nodes = self.cfg["behavior"]["max_nodes"]
            beh = build_behavior_batch([skill_md_text], vocab=vocab,
                                        max_nodes=max_nodes, device=self.device)
            out = self.model(
                [skill_md_text],
                max_length=max_len,
                node_ids=beh["node_ids"],
                adj=beh["adj"],
                node_mask=beh["node_mask"],
            )
            vector = out.h_noise[0] if self.representation == "h_noise" else out.R_local[0]
            base_score = float(out.toxic_score[0].item())
            scores = self._soft_score(base_score, vector)
            score = scores["score"]
            verdict = {
                "toxic": bool(score >= self.threshold),
                **scores,
                "detector_score_name": "counterfactual_soft_gate_score",
                "S_anchor":   float(out.S_anchor[0].item()),
                "S_local":    float(out.S_local[0].item()),
                "S_behavior": float(out.S_behavior[0].item()),
                "threshold":  self.threshold,
                "soft_gate_representation": self.representation,
                "soft_gate_formula": "0.75 * base_score_norm + 0.25 * lift_score_norm",
                "sliding_window_enabled": self.sliding_window_enabled,
                "sliding_window": self.sliding_window,
            }
            return self._attach_evidence(
                verdict,
                skill_md_text=skill_md_text,
                out=out,
                max_length=max_len,
                vocab=vocab,
                max_nodes=max_nodes,
            )
