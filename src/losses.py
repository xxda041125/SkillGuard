"""Binary supervision, paired residual margin and direction separation."""
import torch
import torch.nn.functional as F


def classification_loss(toxic_score: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    p = toxic_score.clamp(eps, 1.0 - eps)
    return F.binary_cross_entropy(p, y.float())


def paired_residual_margin_loss(
    clean_z: torch.Tensor,
    attack_z: torch.Tensor,
    clean_target_norm: float = 0.0,
    margin: float = 1.0,
    clean_weight: float = 1.0,
    shift_weight: float = 1.0,
) -> torch.Tensor:
    """Single residual regularizer for the compact residual recipe.

    The first term keeps clean samples near the residual null point. The second
    term requires attacked variants to move at least ``margin`` away from their
    paired clean counterpart, without rewarding unbounded distance growth.
    """
    clean_norm = clean_z.norm(dim=-1)
    clean_term = F.relu(clean_norm - float(clean_target_norm)).pow(2).mean()
    pair_dist = (attack_z - clean_z).norm(dim=-1)
    shift_term = F.relu(float(margin) - pair_dist).pow(2).mean()
    return float(clean_weight) * clean_term + float(shift_weight) * shift_term


def residual_direction_separation_loss(
    clean_z: torch.Tensor,
    attack_z: torch.Tensor,
    benign_z: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Discourage attack residual shifts from following benign variation.

    ``attack_z - clean_z`` estimates the intervention-induced residual direction.
    ``benign_z - clean_z`` estimates same-domain non-attack variation. The loss
    is bounded because it operates on cosine similarity between normalized
    directions and only penalizes similarity above ``margin``.
    """
    attack_dir = F.normalize(attack_z - clean_z, dim=-1)
    benign_dir = F.normalize(benign_z - clean_z, dim=-1)
    sim = F.cosine_similarity(attack_dir, benign_dir, dim=-1)
    return F.relu(sim - float(margin)).pow(2).mean()
