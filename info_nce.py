import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinglePositiveInfoNCE(nn.Module):
    """
    A single-positive InfoNCE loss for cross-modal alignment:
      - Numerator uses only (i, i) as the "positive" pair
      - Denominator sums over all j that have a *different* label than i
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, audio_features, label_features, class_labels):
        """
        audio_features:  (batch_size, dim)
        label_features:  (batch_size, dim)
        class_labels:    List or tensor of shape (batch_size,) with the class/index/label
        """
        # 1) Normalize feature vectors
        audio_features = F.normalize(audio_features, dim=-1)
        label_features = F.normalize(label_features, dim=-1)
        batch_size = audio_features.shape[0]

        # 2) Build the "positive" mask = identity (i.e. (i,i) is the only 1)
        #    We only want a single positive: the diagonal entry for i == i
        positive_mask = torch.eye(batch_size, device=audio_features.device, dtype=torch.float32)

        # 3) Build the "negative" mask = 1 where labels differ, else 0
        #    i.e. we only want to consider j that have a different label from i
        negative_mask = torch.zeros_like(positive_mask)
        for i in range(batch_size):
            for j in range(batch_size):
                if class_labels[i] != class_labels[j]:
                    negative_mask[i, j] = 1.0

        # 4) Compute cross‐modal logits (dot products / temperature)
        #    label -> audio
        logits_label = (label_features @ audio_features.T) / self.temperature  # (B, B)
        #    audio -> label
        logits_audio = logits_label.T  # (B, B)

        # 5) Exponentiate
        exp_logits_label = logits_label.exp()  # (B, B)
        exp_logits_audio = logits_audio.exp()  # (B, B)

        # 6) Numerator = sum of positives (here it's just the (i,i) diagonal)
        #    We do .sum(dim=1) but since the mask is diagonal, that effectively picks (i,i).
        pos_label = (exp_logits_label * positive_mask).mean(dim=1)  # shape (B,)
        pos_audio = (exp_logits_audio * positive_mask).mean(dim=1)  # shape (B,)

        # 7) Denominator = sum over negatives only
        #    i.e. for row i, sum over columns j where negative_mask[i,j] == 1
        neg_label = (exp_logits_label * negative_mask).sum(dim=1)  # shape (B,)
        neg_audio = (exp_logits_audio * negative_mask).sum(dim=1)  # shape (B,)

        # 8) Probability that (i,i) is correct under i->label or label->i
        p_label = pos_label / (neg_label + 1e-8)
        p_audio = pos_audio / (neg_audio + 1e-8)

        # 9) Final InfoNCE loss
        loss_label = -torch.log(p_label + 1e-8)
        loss_audio = -torch.log(p_audio + 1e-8)
        total_loss = (loss_label + loss_audio).mean()

        return total_loss


