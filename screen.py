"""Fit the frozen reference memory and score visible skill text."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from gate import CounterfactualSoftGateDetector, DetectorGate


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(args):
    return {"checkpoint": digest(args.checkpoint), "config": digest(args.config)}


def partition(manifest, root, split):
    keep = set(manifest["splits"][split]["skill_ids"])
    entries = []
    for label, field in enumerate(("clean", "attacked")):
        for row in manifest[field]:
            identity = row["group"] + "/" + row["skill_name"]
            if identity in keep:
                entries.append(dict(row, path=root / row["path"], identity=identity, label=label))
    return entries


def extrema(values):
    return {"min": float(min(values)), "max": float(max(values))}


def fit(gate, args):
    data = gate.cfg["data"]
    root = args.config.resolve().parent / data["dataset_root"]
    manifest_path = root / data["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    rows, vecs = gate._score_entries(partition(manifest, root, "train"))
    vecs = F.normalize(vecs, dim=-1).to(gate.device)
    clean = {row["identity"]: i for i, row in enumerate(rows) if row["label"] == 0}
    attacks = [i for i, row in enumerate(rows) if row["label"] == 1]
    gate.attack_bank = vecs[attacks]
    gate.paired_clean_bank = vecs[[clean[rows[i]["identity"]] for i in attacks]]
    lift = []
    for chunk in vecs.split(64):
        paired = chunk @ gate.attack_bank.T - chunk @ gate.paired_clean_bank.T
        lift.extend(paired.amax(dim=1).tolist())
    gate.compact_stats = extrema([row["base_score"] for row in rows])
    gate.lift_stats = extrema(lift)

    val, vectors = gate._score_entries(partition(manifest, root, "val"))
    scores = [gate._soft_score(row["base_score"], z.to(gate.device))["score"]
              for row, z in zip(val, vectors)]
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve([r["label"] for r in val], scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]).clip(min=1e-12)
    threshold = float(thresholds[f1.argmax()])
    bundle = dict(signature(args), manifest=digest(manifest_path), threshold=threshold,
                  base_stats=gate.compact_stats, lift_stats=gate.lift_stats,
                  attack=gate.attack_bank.cpu(), clean=gate.paired_clean_bank.cpu())
    args.bank.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.bank)
    print(json.dumps({"pairs": len(attacks), "validation_documents": len(val), "threshold": threshold}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "score", "base"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("detector_config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--threshold", type=float, help="Decision threshold for base-only scoring")
    parser.add_argument("--evidence", action="store_true")
    args = parser.parse_args()
    if args.action != "fit" and args.input is None:
        parser.error("scoring requires --input")
    if args.action == "base":
        if args.threshold is None:
            parser.error("base scoring requires --threshold")
        gate = DetectorGate(args.checkpoint, args.config, args.threshold, args.evidence)
    else:
        if args.bank is None:
            parser.error("paired calibration requires --bank")
        gate = CounterfactualSoftGateDetector(args.checkpoint, args.config, include_evidence=args.evidence)
        gate.batch_size = gate.cfg["data"]["batch_size"]
        gate.representation = "h_noise"
        if args.action == "fit":
            fit(gate, args)
            return
        state = torch.load(args.bank, map_location=gate.device, weights_only=True)
        if any(state[k] != v for k, v in signature(args).items()):
            raise ValueError("Use the checkpoint and configuration that produced the reference bank.")
        gate.attack_bank, gate.paired_clean_bank = state["attack"], state["clean"]
        gate.compact_stats, gate.lift_stats = state["base_stats"], state["lift_stats"]
        gate.threshold = state["threshold"]
    print(json.dumps(gate(args.input.read_text(encoding="utf-8-sig")), indent=2))


if __name__ == "__main__":
    main()
