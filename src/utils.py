"""Shared utilities: behavior-graph batching, device selection, logging."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import yaml

from .data.behavior_extractor import extract_behavior_graph, encode_action_id


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_device(setting: str = "auto") -> torch.device:
    """Select the runtime device.

    - "cpu": force CPU.
    - "cuda": force CUDA.
    - "auto": prefer GPU if available, otherwise CPU.
    """
    if setting == "cpu":
        return torch.device("cpu")
    if setting == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger(log_dir: Path | str | None = None, name: str = "toxic_skill") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / "train.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def build_behavior_batch(
    texts: list[str],
    vocab: list[str],
    max_nodes: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Returns padded (node_ids, adj, node_mask) tensors for a batch of texts."""
    graphs = [extract_behavior_graph(t, vocab=vocab, max_nodes=max_nodes) for t in texts]
    B = len(graphs)
    n_max = max(max(len(g.nodes), 1) for g in graphs)
    n_max = min(n_max, max_nodes)

    node_ids = torch.zeros(B, n_max, dtype=torch.long)
    adj = torch.zeros(B, n_max, n_max, dtype=torch.float32)
    mask = torch.zeros(B, n_max, dtype=torch.float32)

    for b, g in enumerate(graphs):
        nodes = g.nodes[:n_max]
        for i, name in enumerate(nodes):
            node_ids[b, i] = encode_action_id(name, vocab)
            mask[b, i] = 1.0
        for (u, v) in g.edges:
            if u < n_max and v < n_max:
                adj[b, u, v] = 1.0

    return {
        "node_ids": node_ids.to(device),
        "adj": adj.to(device),
        "node_mask": mask.to(device),
    }


def save_checkpoint(model: torch.nn.Module, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))
