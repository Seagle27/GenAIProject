import torch
import torch.nn as nn
import torch.nn.functional as F


class SinglePositiveInfoNCE(nn.Module):
    """
    A single-positive InfoNCE loss for cross-modal alignment:
      - Numerator uses only (i, i) as the "positive" pair
      - Denominator includes (i,i) plus all j that have a *different* label than i
        (so same-label items j != i are excluded altogether).
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, audio_embeddings, text_embeddings, labels):
        """
        audio_features:  (batch_size, dim)
        label_features:  (batch_size, dim)
        class_labels:    List or tensor of shape (batch_size,) with the class/index/label
        """
        # Normalize embeddings to unit norm
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Compute similarity matrix (B, B) and scale by temperature
        similarity_matrix = torch.matmul(audio_embeddings, text_embeddings.T) / self.temperature

        # Expand labels to create a (B, B) matrix for comparison
        labels = labels.unsqueeze(1)  # [B, 1]
        mask = labels == labels.T  # [B, B] - True where same class, False otherwise

        # Mask out self-matching samples
        mask.fill_diagonal_(False)  # Ensure diagonal (self-contrast) is False

        # Apply mask by setting negatives to -inf
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        # Compute log-softmax along rows
        log_probs = F.log_softmax(similarity_matrix, dim=1)

        # Extract diagonal elements (positive pairs) and compute mean loss
        return -log_probs.diag().mean()


class AudioAudioInfoNCESymmetric(nn.Module):
    """
    For each anchor i, the loss is defined as:
      L_i = -log ( sum_{j in P(i)} exp(s_ij/t) / sum_{k != i} exp(s_ik/t) )
    where P(i) contains all indices j (excluding i) with the same label as i.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, audio_embeddings, labels):
        """
        Args:
            audio_embeddings: Tensor of shape (batch_size, embed_dim)
            labels:           Tensor of shape (batch_size,) with integer class labels

        Returns:
            A scalar loss value.
        """
        batch_size = audio_embeddings.size(0)
        if batch_size < 2:
            # Not enough samples for contrastive learning
            return torch.tensor(0.0, device=audio_embeddings.device, requires_grad=True)

        # Normalize the embeddings to unit norm.
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)

        # Compute the similarity matrix and scale by temperature.
        # Shape: (B, B) where s_ij = (audio_i · audio_j) / temperature.
        sim_matrix = torch.matmul(audio_embeddings, audio_embeddings.T) / self.temperature

        # Remove self-similarity from both numerator and denominator.
        # We mask the diagonal by setting it to -infinity so that exp(-inf) becomes 0.
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix.masked_fill(diag_mask, float('-inf'))

        # Exponentiate the similarity scores.
        exp_sim = torch.exp(sim_matrix)

        # Denominator: sum over all non-self similarities for each anchor.
        denom = exp_sim.sum(dim=1)  # Shape: (B,)

        # Build the positive mask: positives are the pairs with the same label.
        # Since self-similarities are already removed, we need to exclude them explicitly.
        labels = labels.unsqueeze(1)  # shape becomes (B, 1)
        positive_mask = (labels == labels.T).float()
        # Remove self-comparisons by zeroing out the diagonal.
        positive_mask = positive_mask.masked_fill(diag_mask, 0)

        # Numerator: sum over all positives.
        numerator = (exp_sim * positive_mask).sum(dim=1)  # Shape: (B,)

        # Only compute loss for anchors that have at least one positive.
        valid = numerator > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=audio_embeddings.device, requires_grad=True)

        # Compute per-anchor loss as described:
        # L_i = - log(numerator_i / denominator_i)
        loss = -torch.log(numerator[valid] / (denom[valid] + 1e-8))
        return loss.mean()
