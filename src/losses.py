import torch
import torch.nn.functional as F


def contrastive_loss(anchor, positive, negative, margin=0.5):
    """Compute margin ranking loss for contrastive learning"""
    # Compute cosine similarities
    pos_sim = F.cosine_similarity(anchor, positive)
    neg_sim = F.cosine_similarity(anchor, negative)

    # Margin ranking loss: encourage pos_sim > neg_sim + margin
    targets = torch.ones_like(pos_sim)
    loss = F.margin_ranking_loss(pos_sim, neg_sim, targets, margin=margin)

    return loss, pos_sim, neg_sim


def supervised_contrastive_loss(batch_embeddings, batch_labels, temperature=0.07):
    """
    Supervised contrastive loss for classification dataset.
    https://github.com/HobbitLong/SupContrast

    Args:
        batch_embeddings: (N, D) normalized feature vectors
        batch_labels: (N,) class labels (0 for cat, 1 for dog)
        temperature: temperature parameter for scaling similarities

    Returns:
        loss: scalar supervised contrastive loss
    """
    batch_size = batch_embeddings.size(0)

    # Ensure we have at least 2 samples
    if batch_size < 2:
        return torch.tensor(0.0, device=batch_embeddings.device, requires_grad=True)

    # Ensure embeddings are normalized (just to be safe)
    batch_embeddings = F.normalize(batch_embeddings, dim=1)

    # Compute cosine similarity matrix: (N, N)
    sim_matrix = torch.matmul(batch_embeddings, batch_embeddings.T)

    # Clamp similarities to avoid extreme values
    sim_matrix = torch.clamp(sim_matrix, min=-1.0, max=1.0)

    # Divide by temperature
    sim_matrix = sim_matrix / temperature

    # Create masks
    mask = torch.eye(batch_size, device=batch_embeddings.device).bool()

    # Create label comparison matrix
    labels = batch_labels.view(-1, 1)
    label_eq = torch.eq(labels, labels.T)  # (N, N): True where same class
    positives_mask = label_eq & (~mask)  # same class but not self

    # Check if we have valid positives for any anchor
    valid_anchors = positives_mask.sum(dim=1) > 0
    if valid_anchors.sum() == 0:
        # No valid anchors (each sample is the only one of its class)
        return torch.tensor(0.0, device=batch_embeddings.device, requires_grad=True)

    # For numerical stability: subtract max per row (excluding diagonal)
    sim_matrix_masked = sim_matrix.clone()
    sim_matrix_masked.masked_fill_(mask, float("-inf"))
    max_vals, _ = torch.max(sim_matrix_masked, dim=1, keepdim=True)
    max_vals = torch.clamp(max_vals, max=10.0)  # Prevent extreme values
    sim_matrix = sim_matrix - max_vals.detach()

    # Set diagonal to very negative value (instead of -inf)
    sim_matrix.masked_fill_(mask, -1e9)

    # Compute exponentials
    exp_sim = torch.exp(
        torch.clamp(sim_matrix, min=-50, max=50)
    )  # Prevent overflow/underflow

    # Denominator: sum over all other samples (excluding self)
    denom = exp_sim.sum(dim=1)  # (N,)

    # Numerator: sum over positives only
    pos_exp = exp_sim * positives_mask.float()
    pos_sum = pos_exp.sum(dim=1)  # (N,)

    # Compute loss for valid anchors only
    eps = 1e-8
    valid_pos_sum = pos_sum[valid_anchors]
    valid_denom = denom[valid_anchors]

    # Add small epsilon to prevent log(0)
    loss_per_anchor = -torch.log((valid_pos_sum + eps) / (valid_denom + eps))

    # Check for NaN/inf and handle
    loss_per_anchor = torch.clamp(loss_per_anchor, min=0.0, max=100.0)
    loss = loss_per_anchor.mean()

    # Final safety check
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"Warning: NaN/Inf loss detected. Returning zero loss.")
        print(f"pos_sum range: {pos_sum.min():.6f} to {pos_sum.max():.6f}")
        print(f"denom range: {denom.min():.6f} to {denom.max():.6f}")
        print(f"sim_matrix range: {sim_matrix.min():.6f} to {sim_matrix.max():.6f}")
        return torch.tensor(0.0, device=batch_embeddings.device, requires_grad=True)

    return loss
