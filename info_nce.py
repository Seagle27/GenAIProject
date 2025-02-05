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
