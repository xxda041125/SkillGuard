"""Train and evaluate Counterfactual SkillGuard with the paper's training objective."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

EXPERIMENT_DIR = Path(__file__).resolve().parent
CSG_ROOT = EXPERIMENT_DIR.parents[1]
if str(CSG_ROOT) not in sys.path:
    sys.path.insert(0, str(CSG_ROOT))

from src import losses as L  # noqa: E402
from src.data.dataset import FlatToxicSkillDataset, ToxicSkillDataset, flat_collate, paired_collate  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.utils import build_behavior_batch, load_config, save_checkpoint, select_device, setup_logger  # noqa: E402

ROLES_ALL = ("clean", "attacked", "neighbor_clean", "neighbor_attacked")
ROLES_PAIRED = ("clean", "attacked")


def _resolve_path(raw: str | Path | None, base: Path) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.epochs is not None:
        cfg["optim"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = int(args.batch_size)
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    if args.output_dir is not None:
        cfg["runtime"]["output_dir"] = args.output_dir
    if args.init_checkpoint is not None:
        cfg.setdefault("runtime", {})["init_checkpoint"] = args.init_checkpoint
    if args.recipe is not None:
        cfg.setdefault("losses", {})["scheme"] = args.recipe
    if args.manifest is not None:
        cfg["data"]["manifest"] = args.manifest


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flatten_batch(batch: dict[str, Any], classification_roles: str) -> tuple[list[str], torch.Tensor, dict[str, slice]]:
    roles = ROLES_ALL if classification_roles == "all" else ROLES_PAIRED
    texts: list[str] = []
    slices: dict[str, slice] = {}
    cursor = 0
    for role in roles:
        role_texts = batch[f"{role}_text"]
        slices[role] = slice(cursor, cursor + len(role_texts))
        texts.extend(role_texts)
        cursor += len(role_texts)
    batch_size = len(batch["clean_text"])
    if roles == ROLES_ALL:
        labels = torch.tensor([0] * batch_size + [1] * batch_size + [0] * batch_size + [1] * batch_size, dtype=torch.long)
    else:
        labels = torch.tensor([0] * batch_size + [1] * batch_size, dtype=torch.long)
    return texts, labels, slices


def _paired_residual_margin_loss_with_stats(
    clean_z: torch.Tensor,
    attack_z: torch.Tensor,
    *,
    clean_target_norm: float = 0.0,
    margin: float = 1.0,
    clean_weight: float = 1.0,
    shift_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Official residual loss plus training diagnostics."""
    clean_norm = clean_z.norm(dim=-1)
    pair_dist = (attack_z - clean_z).norm(dim=-1)
    loss = L.paired_residual_margin_loss(
        clean_z,
        attack_z,
        clean_target_norm=clean_target_norm,
        margin=margin,
        clean_weight=clean_weight,
        shift_weight=shift_weight,
    )
    stats = {
        "pair_dist_mean": float(pair_dist.detach().cpu().mean().item()),
        "clean_norm_mean": float(clean_norm.detach().cpu().mean().item()),
    }
    return loss, stats


def _residual_direction_loss_with_stats(
    clean_z: torch.Tensor,
    attack_z: torch.Tensor,
    benign_z: torch.Tensor,
    *,
    margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Optional ablation loss for separating attack and benign directions."""
    attack_dir = torch.nn.functional.normalize(attack_z - clean_z, dim=-1)
    benign_dir = torch.nn.functional.normalize(benign_z - clean_z, dim=-1)
    sim = torch.nn.functional.cosine_similarity(attack_dir, benign_dir, dim=-1)
    loss = L.residual_direction_separation_loss(
        clean_z,
        attack_z,
        benign_z,
        margin=margin,
    )
    stats = {
        "direction_cos_mean": float(sim.detach().cpu().mean().item()),
    }
    return loss, stats


def _limit_dataset(ds: Any, max_items: int | None, seed: int) -> Any:
    if max_items is None or max_items <= 0 or len(ds) <= max_items:
        return ds
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    return Subset(ds, sorted(indices[:max_items]))


def _limit_flat_balanced(ds: FlatToxicSkillDataset, max_items: int | None, seed: int) -> Any:
    if max_items is None or max_items <= 0 or len(ds) <= max_items:
        return ds
    by_label: dict[int, list[int]] = {0: [], 1: []}
    for idx, item in enumerate(ds.items):
        by_label[int(item.label)].append(idx)
    rng = random.Random(seed)
    for values in by_label.values():
        rng.shuffle(values)
    half = max_items // 2
    selected = by_label[0][:half] + by_label[1][: max_items - half]
    if len(selected) < max_items:
        remaining = [i for i in range(len(ds)) if i not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_items - len(selected)])
    return Subset(ds, sorted(selected))


def _make_flat_loader(
    cfg: dict[str, Any],
    dataset_root: Path,
    args: argparse.Namespace,
    split: str,
    seed_offset: int,
) -> DataLoader:
    seed = int(cfg["data"]["seed"])
    ds = FlatToxicSkillDataset(
        dataset_root=dataset_root,
        manifest_name=cfg["data"]["manifest"],
        split=split,
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        seed=seed,
    )
    ds = _limit_flat_balanced(ds, args.max_eval_items, seed + seed_offset)
    return DataLoader(
        ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=False,
        collate_fn=flat_collate,
        num_workers=int(cfg["data"]["num_workers"]),
    )


def _make_loaders(cfg: dict[str, Any], dataset_root: Path, args: argparse.Namespace) -> tuple[DataLoader, DataLoader, dict[str, DataLoader]]:
    seed = int(cfg["data"]["seed"])
    train_ds = ToxicSkillDataset(
        dataset_root=dataset_root,
        manifest_name=cfg["data"]["manifest"],
        split="train",
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        seed=seed,
    )
    train_ds = _limit_dataset(train_ds, args.max_train_pairs, seed)

    batch_size = int(cfg["data"]["batch_size"])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=paired_collate,
        num_workers=int(cfg["data"]["num_workers"]),
    )
    val_loader = _make_flat_loader(cfg, dataset_root, args, split="val", seed_offset=1)

    eval_splits = list(dict.fromkeys(cfg["data"].get("eval_splits", ["test"])))
    if not eval_splits:
        eval_splits = ["test"]
    eval_loaders = {
        split: _make_flat_loader(cfg, dataset_root, args, split=split, seed_offset=2 + idx)
        for idx, split in enumerate(eval_splits)
    }
    return train_loader, val_loader, eval_loaders


def _ensure_manifest_readable(cfg: dict[str, Any], dataset_root: Path, output_dir: Path) -> None:
    """Dataset code reads manifests as utf-8; normalize BOM manifests locally."""
    manifest_path = _resolve_path(cfg["data"]["manifest"], dataset_root)
    if manifest_path is None:
        raise ValueError("data.manifest is required")
    try:
        payload = manifest_path.read_text(encoding="utf-8")
        if not payload.startswith("\ufeff"):
            return
    except UnicodeError:
        payload = manifest_path.read_text(encoding="utf-8-sig")
    else:
        payload = payload.lstrip("\ufeff")

    normalized = output_dir / f"{manifest_path.stem}.utf8.json"
    normalized.write_text(payload, encoding="utf-8")
    cfg["data"]["manifest"] = str(normalized)


def _model_forward(model: torch.nn.Module, texts: list[str], cfg: dict[str, Any], device: torch.device):
    max_len = int(cfg["data"]["max_token_length"])
    vocab = cfg["behavior"]["action_vocab"]
    max_nodes = int(cfg["behavior"]["max_nodes"])
    beh = build_behavior_batch(texts, vocab=vocab, max_nodes=max_nodes, device=device)
    return model(
        texts,
        max_length=max_len,
        node_ids=beh["node_ids"],
        adj=beh["adj"],
        node_mask=beh["node_mask"],
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    cfg: dict[str, Any],
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    loss_cfg = cfg["losses"]
    scheme = str(loss_cfg.get("scheme", "two_objective")).lower()
    grad_clip = float(cfg["optim"]["grad_clip"])
    classification_roles = str(loss_cfg.get("classification_roles", "all")).lower()
    if classification_roles not in {"all", "paired"}:
        raise ValueError("losses.classification_roles must be 'all' or 'paired'")
    w_direction = float(loss_cfg.get("w_direction", 0.0))
    if w_direction > 0.0 and classification_roles != "all":
        raise ValueError("direction separation requires losses.classification_roles='all'")

    running: dict[str, float] = {
        "loss": 0.0,
        "cls": 0.0,
        "residual": 0.0,
        "direction": 0.0,
        "pair_dist_mean": 0.0,
        "clean_norm_mean": 0.0,
        "direction_cos_mean": 0.0,
    }
    n_steps = 0

    for batch in tqdm(loader, desc=f"epoch {epoch} train", leave=False):
        texts, labels, slices = _flatten_batch(batch, classification_roles=classification_roles)
        labels = labels.to(device)
        out = _model_forward(model, texts, cfg, device)

        l_cls = L.classification_loss(out.toxic_score, labels)

        if scheme == "two_objective":
            clean_slice = slices["clean"]
            attack_slice = slices["attacked"]
            l_residual, residual_stats = _paired_residual_margin_loss_with_stats(
                out.h_noise[clean_slice],
                out.h_noise[attack_slice],
                clean_target_norm=float(loss_cfg.get("clean_target_norm", 0.0)),
                margin=float(loss_cfg.get("residual_margin", 1.0)),
                clean_weight=float(loss_cfg.get("residual_clean_weight", 1.0)),
                shift_weight=float(loss_cfg.get("residual_shift_weight", 1.0)),
            )
            if w_direction > 0.0:
                neighbor_clean_slice = slices["neighbor_clean"]
                l_direction, direction_stats = _residual_direction_loss_with_stats(
                    out.h_noise[clean_slice],
                    out.h_noise[attack_slice],
                    out.h_noise[neighbor_clean_slice],
                    margin=float(loss_cfg.get("direction_margin", 0.0)),
                )
            else:
                l_direction = out.h_noise.new_tensor(0.0)
                direction_stats = {"direction_cos_mean": 0.0}
            loss = (
                float(loss_cfg["w_cls"]) * l_cls
                + float(loss_cfg["w_residual"]) * l_residual
                + w_direction * l_direction
            )
        else:
            raise ValueError(f"Main training entry only supports two_objective, got: {scheme}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running["loss"] += float(loss.detach().cpu().item())
        running["cls"] += float(l_cls.detach().cpu().item())
        running["residual"] += float(l_residual.detach().cpu().item())
        running["direction"] += float(l_direction.detach().cpu().item())
        for key in ("pair_dist_mean", "clean_norm_mean"):
            running[key] += float(residual_stats.get(key, 0.0))
        running["direction_cos_mean"] += float(direction_stats.get("direction_cos_mean", 0.0))
        n_steps += 1

    return {key: value / max(n_steps, 1) for key, value in running.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
    fixed_threshold: float | None = None,
) -> dict[str, Any]:
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        out = _model_forward(model, batch["text"], cfg, device)
        scores.extend(float(x) for x in out.toxic_score.detach().cpu().tolist())
        labels.extend(int(x) for x in batch["label"].cpu().tolist())
    return metrics_from_scores(scores, labels, fixed_threshold=fixed_threshold)


def metrics_from_scores(
    scores: list[float],
    labels: list[int],
    fixed_threshold: float | None = None,
) -> dict[str, Any]:
    pairs = list(zip(scores, labels))
    if not pairs:
        return {}
    thresholds = [i / 100 for i in range(5, 96, 5)]
    best = {"threshold": 0.5, "f1": 0.0, "acc": 0.0, "precision": 0.0, "recall": 0.0}
    for threshold in thresholds:
        stats = confusion(scores, labels, threshold)
        if stats["f1"] > best["f1"]:
            best = {key: stats[key] for key in ("threshold", "f1", "acc", "precision", "recall")}

    clean_scores = [score for score, label in pairs if label == 0]
    attack_scores = [score for score, label in pairs if label == 1]
    fixed_fpr = {}
    for target in (0.01, 0.05, 0.10, 0.20):
        threshold = threshold_for_clean_fpr(clean_scores, target)
        stats = confusion(scores, labels, threshold)
        fixed_fpr[str(target)] = stats

    metrics = {
        "n": len(pairs),
        "n_clean": len(clean_scores),
        "n_attack": len(attack_scores),
        "best_f1": best,
        "fixed_clean_fpr": fixed_fpr,
        "mean_score_clean": sum(clean_scores) / max(len(clean_scores), 1),
        "mean_score_attack": sum(attack_scores) / max(len(attack_scores), 1),
    }
    if fixed_threshold is not None:
        metrics["threshold_selected_on_val"] = confusion(scores, labels, fixed_threshold)
    try:
        from sklearn.metrics import roc_auc_score

        if len(set(labels)) == 2:
            metrics["auroc"] = float(roc_auc_score(labels, scores))
    except Exception:
        pass
    return metrics


def threshold_for_clean_fpr(clean_scores: list[float], max_clean_fpr: float) -> float:
    if not clean_scores:
        return 0.5
    ranked = sorted(clean_scores, reverse=True)
    allowed_fp = int(max_clean_fpr * len(ranked))
    if allowed_fp <= 0:
        return ranked[0] + 1e-6
    if allowed_fp >= len(ranked):
        return ranked[-1]
    return ranked[allowed_fp - 1]


def confusion(scores: list[float], labels: list[int], threshold: float) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        pred = 1 if score >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    clean_fpr = fp / max(fp + tn, 1)
    return {
        "threshold": float(threshold),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "acc": float(acc),
        "clean_fpr": float(clean_fpr),
    }


def build_optimizer_and_scheduler(model: torch.nn.Module, cfg: dict[str, Any], steps_per_epoch: int):
    epochs = int(cfg["optim"]["epochs"])
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, int(total_steps * float(cfg["optim"]["warmup_ratio"])))
    head_lr = float(cfg["optim"]["lr"])
    backbone_lr = float(cfg["optim"].get("backbone_lr", head_lr))
    weight_decay = float(cfg["optim"]["weight_decay"])

    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder.backbone."):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)
    param_groups = [{"params": head_params, "lr": head_lr, "weight_decay": weight_decay}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})
    optimizer = torch.optim.AdamW(param_groups)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_two_objective.yaml")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="train")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for eval mode, or init checkpoint when --mode eval.")
    parser.add_argument("--init-checkpoint", default=None, help="Optional checkpoint to initialize before training.")
    parser.add_argument("--recipe", choices=["two_objective"], default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument(
        "--train-with-sliding-window",
        action="store_true",
        help="Keep sliding_window.enabled during training. By default training disables it; runtime inference can still use sliding.",
    )
    args = parser.parse_args()

    config_path = _resolve_path(args.config, EXPERIMENT_DIR)
    if config_path is None:
        raise ValueError("config path is required")
    cfg = load_config(config_path)
    _apply_overrides(cfg, args)
    if args.mode in {"train", "both"} and not args.train_with_sliding_window:
        cfg.setdefault("sliding_window", {})["enabled"] = False
    _set_seed(int(cfg["data"]["seed"]))

    dataset_root = _resolve_path(cfg["data"]["dataset_root"], config_path.parent)
    if dataset_root is None:
        raise ValueError("data.dataset_root is required")
    output_dir = _resolve_path(cfg["runtime"]["output_dir"], config_path.parent)
    if output_dir is None:
        raise ValueError("runtime.output_dir is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_manifest_readable(cfg, dataset_root, output_dir)

    logger = setup_logger(output_dir, name=f"two_objective_{output_dir.name}")
    logger.info(f"config: {config_path}")
    logger.info(f"dataset_root: {dataset_root}")
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"loss_scheme: {cfg['losses'].get('scheme')}")

    device = select_device(str(cfg["runtime"].get("device", "auto")))
    logger.info(f"device: {device}")

    train_loader, val_loader, eval_loaders = _make_loaders(cfg, dataset_root, args)
    eval_batch_summary = ", ".join(f"{name}: {len(loader)}" for name, loader in eval_loaders.items())
    logger.info(f"train batches: {len(train_loader)} | val batches: {len(val_loader)} | eval batches: {eval_batch_summary}")

    model = build_model(cfg).to(device)
    init_checkpoint = args.checkpoint if args.mode == "eval" and args.checkpoint else cfg.get("runtime", {}).get("init_checkpoint")
    if init_checkpoint:
        init_path = _resolve_path(init_checkpoint, config_path.parent)
        if init_path is None:
            raise ValueError("checkpoint path could not be resolved")
        state = torch.load(str(init_path), map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"], strict=True)
        logger.info(f"loaded checkpoint: {init_path}")

    effective_config = {
        "config": cfg,
        "max_train_pairs": args.max_train_pairs,
        "max_eval_items": args.max_eval_items,
    }
    (output_dir / "effective_config.json").write_text(json.dumps(effective_config, indent=2), encoding="utf-8")

    history: list[dict[str, Any]] = []
    if args.mode in {"train", "both"}:
        optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, len(train_loader))
        best_val_f1 = -1.0
        for epoch in range(1, int(cfg["optim"]["epochs"]) + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, cfg, device, epoch)
            val_metrics = evaluate(model, val_loader, cfg, device)
            row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            history.append(row)
            val_f1 = float(val_metrics.get("best_f1", {}).get("f1", 0.0))
            logger.info(
                "epoch %s | train_loss=%.4f cls=%.4f residual=%.4f direction=%.4f | val_f1=%.4f auroc=%.4f",
                epoch,
                train_metrics["loss"],
                train_metrics["cls"],
                train_metrics["residual"],
                train_metrics["direction"],
                val_f1,
                float(val_metrics.get("auroc", 0.0)),
            )
            if bool(cfg["runtime"].get("save_every_epoch", True)):
                save_checkpoint(model, output_dir / f"ckpt_epoch{epoch}.pt", extra={"cfg": cfg, "val": val_metrics})
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                save_checkpoint(model, output_dir / "ckpt_best.pt", extra={"cfg": cfg, "val": val_metrics})
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    eval_metrics = {}
    if args.mode in {"eval", "both"}:
        if args.mode == "both":
            best = torch.load(output_dir / "ckpt_best.pt", map_location=device, weights_only=True)
            model.load_state_dict(best["state_dict"])
        eval_metrics["val"] = evaluate(model, val_loader, cfg, device)
        val_threshold = float(eval_metrics["val"].get("best_f1", {}).get("threshold", 0.5))
        eval_metrics["threshold_source"] = {
            "split": "val",
            "metric": "best_f1.threshold",
            "threshold": val_threshold,
        }
        for split, loader in eval_loaders.items():
            eval_metrics[split] = evaluate(model, loader, cfg, device, fixed_threshold=val_threshold)
        (output_dir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
        for split in eval_loaders:
            metrics = eval_metrics[split]
            logger.info(
                "%s | best_f1=%.4f val_threshold_f1=%.4f auroc=%.4f mean_clean=%.4f mean_attack=%.4f",
                split,
                float(metrics.get("best_f1", {}).get("f1", 0.0)),
                float(metrics.get("threshold_selected_on_val", {}).get("f1", 0.0)),
                float(metrics.get("auroc", 0.0)),
                float(metrics.get("mean_score_clean", 0.0)),
                float(metrics.get("mean_score_attack", 0.0)),
            )


if __name__ == "__main__":
    main()
