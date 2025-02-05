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


class AudioAudioInfoNCE(nn.Module):
    """
    InfoNCE loss for aligning audio embeddings.
    For each anchor embedding, all other embeddings with the same label are
    considered positives. The loss is computed as the negative log probability
    assigned to the positive pairs when the similarity scores (excluding self)
    are passed through a softmax.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, audio_embeddings, labels):
        """
        audio_embeddings: Tensor of shape (batch_size, embed_dim)
        labels:           Tensor of shape (batch_size,) with integer class labels

        Returns:
          A scalar loss value.
        """
        # Normalize the embeddings to have unit norm.
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)

        # Compute the similarity matrix [B, B] scaled by the temperature.
        sim_matrix = torch.matmul(audio_embeddings, audio_embeddings.T) / self.temperature

        batch_size = sim_matrix.size(0)
        # Create a boolean mask for the diagonal (self-similarity) and set them to -inf.
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix.masked_fill_(diag_mask, float('-inf'))

        # Compute the log-softmax over each row.
        log_probs = F.log_softmax(sim_matrix, dim=1)

        # Create a mask for positive pairs:
        # Two samples are positive if they have the same label.
        # (We already excluded self-similarities above.)
        labels = labels.unsqueeze(1)  # [B, 1]
        positive_mask = (labels == labels.T)
        positive_mask.masked_fill_(diag_mask, False)  # ensure self-pairs are not used

        # Compute the loss:
        # For each anchor i, average the negative log probability over all positive pairs.
        # We sum over all positive pairs and divide by the total number of positive pairs.
        num_positives = positive_mask.float().sum()
        loss = - (log_probs * positive_mask.float()).sum() / (num_positives + 1e-8)

        return loss
