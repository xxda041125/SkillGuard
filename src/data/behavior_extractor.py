"""Heuristic behavior graph extractor.

Builds a small directed graph G=(V,E) for each Skill text where:
- Each node V_i is one occurrence of an "atomic operation" verb from a fixed
  vocabulary (read / write / download / execute / send / upload / ...).
- Edges encode sequential execution dependency: an edge (v_i -> v_j) is added
  when v_i appears before v_j inside the same code block, OR when they are
  adjacent verbs in the linear flow of the markdown.

This is a lightweight stand-in for an LLM-based extractor. It captures:
- The presence/absence of dangerous operation chains
- Simple ordering (download -> execute, read -> upload, etc.)

Output schema (per skill):
    nodes: list[str]  (length n, names from action vocab)
    edges: list[tuple[int,int]]  (directed)
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class BehaviorGraph:
    nodes: list[str]
    edges: list[tuple[int, int]]
    spans: list[tuple[int, int]]  # (start_char, end_char) of each node match


_DEFAULT_VOCAB = [
    "read", "write", "download", "upload", "execute",
    "fetch", "send", "install", "run", "delete",
    "encode", "decode", "compress", "decrypt", "request",
    "export", "source",
]

# Aliases that map to canonical actions
_ALIASES = {
    "curl": "download",
    "wget": "download",
    "git clone": "download",
    "pip install": "install",
    "uv pip install": "install",
    "npm install": "install",
    "post": "send",
    "get": "fetch",
    "open": "read",
    "cat": "read",
    "rm": "delete",
    "remove": "delete",
    "bash": "execute",
    "sh ": "execute",
    "python ": "execute",
    "subprocess": "execute",
    "os.system": "execute",
    "eval(": "execute",
    "exec(": "execute",
    "base64": "encode",
}


def _build_pattern(vocab: list[str]) -> re.Pattern:
    tokens = sorted(set(vocab) | set(_ALIASES.keys()), key=len, reverse=True)
    escaped = [re.escape(t) for t in tokens]
    pat = r"(?<![A-Za-z0-9_])(" + "|".join(escaped) + r")"
    return re.compile(pat, re.IGNORECASE)


def extract_behavior_graph(text: str, vocab: list[str] | None = None,
                            max_nodes: int = 16) -> BehaviorGraph:
    vocab = vocab or _DEFAULT_VOCAB
    pat = _build_pattern(vocab)

    nodes: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in pat.finditer(text):
        raw = m.group(0).lower()
        canon = _ALIASES.get(raw, raw)
        if canon not in vocab:
            # Map unknowns to nearest default action
            continue
        nodes.append(canon)
        spans.append((m.start(), m.end()))
        if len(nodes) >= max_nodes:
            break

    if not nodes:
        # Always emit a single "read" placeholder so the GNN has at least one node.
        return BehaviorGraph(nodes=["read"], edges=[], spans=[(0, 0)])

    # Edges: linear chain plus same-codeblock edges.
    edges: list[tuple[int, int]] = []

    # 1) sequential edge between consecutive nodes
    for i in range(len(nodes) - 1):
        edges.append((i, i + 1))

    # 2) extra edges between nodes inside the same fenced code block
    code_blocks = _find_code_block_spans(text)
    for (cb_start, cb_end) in code_blocks:
        idxs = [i for i, (s, e) in enumerate(spans) if cb_start <= s < cb_end]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                edge = (idxs[a], idxs[b])
                if edge not in edges:
                    edges.append(edge)

    return BehaviorGraph(nodes=nodes, edges=edges, spans=spans)


def _find_code_block_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    while True:
        start = text.find("```", i)
        if start == -1:
            break
        end = text.find("```", start + 3)
        if end == -1:
            break
        spans.append((start, end))
        i = end + 3
    return spans


def encode_action_id(action: str, vocab: list[str]) -> int:
    try:
        return vocab.index(action)
    except ValueError:
        return 0
