"""ToxicSkill dataset loader.

Reads new_dataset/manifest.json (clean + attacked entries), pairs each attacked
skill with its clean counterpart by skill_name, and yields training examples
suitable for the ToxicSkill v3+ losses (paired clean/attacked + functional
neighbor pairs).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


@dataclass
class SkillEntry:
    skill_name: str
    group: str
    path: Path
    label: int                # 0 = clean, 1 = attacked
    attack_type: Optional[str] = None
    generator_id: Optional[str] = None
    text: str = ""


def _read_skill_md(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _entry_skill_id(entry: SkillEntry) -> str:
    return f"{entry.group}/{entry.skill_name}"


def _stable_split_offset(split: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(split)) % 1000


def _manifest_split_spec(manifest: dict, split: str) -> Optional[dict]:
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        return None
    spec = splits.get(split)
    if spec is None:
        return None
    if isinstance(spec, list):
        return {"skill_ids": spec}
    if isinstance(spec, dict):
        return spec
    raise ValueError(f"Invalid manifest split spec for {split}: {type(spec).__name__}")


def _split_skill_ids_from_manifest(spec: dict) -> set[str]:
    raw_ids = spec.get("skill_ids")
    if raw_ids is None:
        raw_ids = spec.get("skills")
    if raw_ids is None:
        raise ValueError("Manifest split spec must define skill_ids")
    return {str(value) for value in raw_ids}


def _split_generators_from_manifest(spec: dict) -> Optional[set[str]]:
    generators = spec.get("generators")
    if generators is None:
        return None
    return {str(value) for value in generators}


def _entry_in_skill_ids(entry: SkillEntry, skill_ids: set[str]) -> bool:
    # Prefer the paper protocol key, but accept legacy skill_name-only lists.
    return _entry_skill_id(entry) in skill_ids or entry.skill_name in skill_ids


class ToxicSkillDataset(Dataset):
    """Returns paired examples for the ToxicSkill v3+ training loop.

    Each item is a tuple of:
        anchor_clean: SkillEntry  (label=0)
        anchor_attacked: SkillEntry (label=1, same skill_name as anchor_clean)
        neighbor_clean: SkillEntry (different skill_name, same group, label=0)
        neighbor_attacked: SkillEntry (different skill_name, same group, label=1)

    Iteration is driven by the attacked-skill list because every attacked entry
    has exactly one clean counterpart.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        manifest_name: str = "manifest.json",
        split: str = "train",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.root = Path(dataset_root).resolve()
        manifest_path = self.root / manifest_name
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        clean_entries: list[SkillEntry] = []
        attacked_entries: list[SkillEntry] = []

        for item in manifest.get("clean", []):
            clean_entries.append(
                SkillEntry(
                    skill_name=item["skill_name"],
                    group=item["group"],
                    path=self.root / item["path"],
                    label=0,
                    generator_id=item.get("generator_id"),
                )
            )
        for item in manifest.get("attacked", []):
            attacked_entries.append(
                SkillEntry(
                    skill_name=item["skill_name"],
                    group=item["group"],
                    path=self.root / item["path"],
                    label=1,
                    attack_type=item.get("attack_type"),
                    generator_id=item.get("generator_id"),
                )
            )

        # Index clean by original skill id for pairing. The canonical v3 manifest keeps
        # augmented clean variants outside the clean pool, so paired training
        # uses original clean skills as the semantic baseline.
        clean_by_id: dict[str, list[SkillEntry]] = defaultdict(list)
        for e in clean_entries:
            clean_by_id[_entry_skill_id(e)].append(e)

        # Keep only attacked entries that have a matching clean entry
        attacked_entries = [e for e in attacked_entries if _entry_skill_id(e) in clean_by_id]

        explicit_split = _manifest_split_spec(manifest, split)
        if explicit_split is not None:
            keep_ids = _split_skill_ids_from_manifest(explicit_split)
            keep_generators = _split_generators_from_manifest(explicit_split)
            clean_entries = [e for e in clean_entries if _entry_in_skill_ids(e, keep_ids)]
            attacked_entries = [
                e
                for e in attacked_entries
                if _entry_in_skill_ids(e, keep_ids)
                and (keep_generators is None or (e.generator_id or "") in keep_generators)
            ]
        else:
            # Legacy behavior: split by original skill id to avoid leakage
            # between train, validation, and test.
            rng = random.Random(seed)
            skill_ids = sorted({_entry_skill_id(e) for e in attacked_entries})
            rng.shuffle(skill_ids)
            n = len(skill_ids)
            n_test = int(n * test_ratio)
            n_val = int(n * val_ratio)
            test_ids = set(skill_ids[:n_test])
            val_ids = set(skill_ids[n_test : n_test + n_val])
            train_ids = set(skill_ids[n_test + n_val :])

            if split == "train":
                keep_ids = train_ids
            elif split == "val":
                keep_ids = val_ids
            elif split == "test":
                keep_ids = test_ids
            else:
                raise ValueError(f"Unknown split: {split}")
            clean_entries = [e for e in clean_entries if _entry_skill_id(e) in keep_ids]
            attacked_entries = [e for e in attacked_entries if _entry_skill_id(e) in keep_ids]

        clean_by_id = defaultdict(list)
        for e in clean_entries:
            clean_by_id[_entry_skill_id(e)].append(e)

        # Group skills by group for functional-neighbor sampling
        clean_by_group: dict[str, list[SkillEntry]] = defaultdict(list)
        attacked_by_group: dict[str, list[SkillEntry]] = defaultdict(list)
        for e in clean_entries:
            clean_by_group[e.group].append(e)
        for e in attacked_entries:
            attacked_by_group[e.group].append(e)

        self.clean_by_id = clean_by_id
        self.clean_by_group = clean_by_group
        self.attacked_by_group = attacked_by_group

        self.split = split
        self.attacked_entries = attacked_entries
        self._rng = random.Random(seed + _stable_split_offset(split))

        # Cache of file contents to avoid repeated reads
        self._text_cache: dict[Path, str] = {}

    def _load_text(self, entry: SkillEntry) -> str:
        if entry.path not in self._text_cache:
            self._text_cache[entry.path] = _read_skill_md(entry.path)
        entry.text = self._text_cache[entry.path]
        return entry.text

    def __len__(self) -> int:
        return len(self.attacked_entries)

    def _sample_neighbor(self, group: str, exclude_skill_id: str, label: int) -> SkillEntry:
        pool = (self.clean_by_group if label == 0 else self.attacked_by_group).get(group, [])
        candidates = [e for e in pool if _entry_skill_id(e) != exclude_skill_id]
        if not candidates:
            raise ValueError(f"No neighbor candidates available for split={self.split} label={label}")
        return self._rng.choice(candidates)

    def __getitem__(self, idx: int) -> dict:
        attacked = self.attacked_entries[idx]
        attacked_skill_id = _entry_skill_id(attacked)
        clean_variants = self.clean_by_id[attacked_skill_id]
        # randomly pick one clean variant per call (orig / paraphrased / extra)
        clean = self._rng.choice(clean_variants) if len(clean_variants) > 1 else clean_variants[0]

        neighbor_attacked = self._sample_neighbor(attacked.group, attacked_skill_id, label=1)
        neighbor_clean = self.clean_by_id[_entry_skill_id(neighbor_attacked)][0]

        for e in (clean, attacked, neighbor_clean, neighbor_attacked):
            self._load_text(e)

        return {
            "clean_text": clean.text,
            "attacked_text": attacked.text,
            "neighbor_clean_text": neighbor_clean.text,
            "neighbor_attacked_text": neighbor_attacked.text,
            "skill_name": attacked.skill_name,
            "skill_id": attacked_skill_id,
            "group": attacked.group,
            "attack_type": attacked.attack_type or "",
            "generator_id": attacked.generator_id or "",
            "neighbor_generator_id": neighbor_attacked.generator_id or "",
        }


class FlatToxicSkillDataset(Dataset):
    """Single-sample dataset for evaluation/inference. Each item is one skill
    (clean or attacked) with a binary label."""

    def __init__(
        self,
        dataset_root: str | Path,
        manifest_name: str = "manifest.json",
        split: str = "test",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        include_clean: bool = True,
        include_attacked: bool = True,
    ) -> None:
        self.root = Path(dataset_root).resolve()
        with (self.root / manifest_name).open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        clean = [
            SkillEntry(skill_name=i["skill_name"], group=i["group"],
                       path=self.root / i["path"], label=0,
                       generator_id=i.get("generator_id"))
            for i in manifest.get("clean", [])
        ]
        attacked = [
            SkillEntry(skill_name=i["skill_name"], group=i["group"],
                       path=self.root / i["path"], label=1,
                       attack_type=i.get("attack_type"),
                       generator_id=i.get("generator_id"))
            for i in manifest.get("attacked", [])
        ]

        explicit_split = _manifest_split_spec(manifest, split)
        if explicit_split is not None:
            keep = _split_skill_ids_from_manifest(explicit_split)
            keep_generators = _split_generators_from_manifest(explicit_split)
        else:
            # Same legacy partition as the paired dataset, keyed by original skill.
            rng = random.Random(seed)
            all_ids = sorted({_entry_skill_id(e) for e in attacked})
            rng.shuffle(all_ids)
            n = len(all_ids)
            n_test = int(n * test_ratio)
            n_val = int(n * val_ratio)
            test_ids = set(all_ids[:n_test])
            val_ids = set(all_ids[n_test : n_test + n_val])
            train_ids = set(all_ids[n_test + n_val :])

            if split == "train":
                keep = train_ids
            elif split == "val":
                keep = val_ids
            elif split == "test":
                keep = test_ids
            else:
                raise ValueError(f"Unknown split: {split}")
            keep_generators = None

        items: list[SkillEntry] = []
        if include_clean:
            items.extend([e for e in clean if _entry_in_skill_ids(e, keep)])
        if include_attacked:
            items.extend(
                [
                    e
                    for e in attacked
                    if _entry_in_skill_ids(e, keep)
                    and (keep_generators is None or (e.generator_id or "") in keep_generators)
                ]
            )

        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        e = self.items[idx]
        text = _read_skill_md(e.path)
        return {
            "text": text,
            "label": e.label,
            "skill_name": e.skill_name,
            "skill_id": _entry_skill_id(e),
            "group": e.group,
            "attack_type": e.attack_type or "",
            "generator_id": e.generator_id or "",
        }


def paired_collate(batch: list[dict]) -> dict:
    """Collate paired examples into lists keyed by role."""
    return {
        "clean_text": [b["clean_text"] for b in batch],
        "attacked_text": [b["attacked_text"] for b in batch],
        "neighbor_clean_text": [b["neighbor_clean_text"] for b in batch],
        "neighbor_attacked_text": [b["neighbor_attacked_text"] for b in batch],
        "skill_name": [b["skill_name"] for b in batch],
        "skill_id": [b["skill_id"] for b in batch],
        "group": [b["group"] for b in batch],
        "attack_type": [b["attack_type"] for b in batch],
        "generator_id": [b["generator_id"] for b in batch],
        "neighbor_generator_id": [b["neighbor_generator_id"] for b in batch],
    }


def flat_collate(batch: list[dict]) -> dict:
    return {
        "text": [b["text"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "skill_name": [b["skill_name"] for b in batch],
        "skill_id": [b["skill_id"] for b in batch],
        "group": [b["group"] for b in batch],
        "attack_type": [b["attack_type"] for b in batch],
        "generator_id": [b["generator_id"] for b in batch],
    }
