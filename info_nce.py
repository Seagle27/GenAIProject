import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InfoNCELoss(nn.Module):
    def __init__(self, init_logit_scale=1.0):
        super(InfoNCELoss, self).__init__()

        self.log_logit_scale = nn.Parameter(torch.tensor(np.log(init_logit_scale), dtype=torch.float32))

    def forward(self, audio_features, label_features, class_labels):
        # Normalize the embeddings.
        audio_features = F.normalize(audio_features, dim=-1)
        label_features = F.normalize(label_features, dim=-1)

        # Create a mask for same-class positives
        batch_size = len(class_labels)
        mask = torch.zeros((batch_size, batch_size), dtype=torch.float32, device=audio_features.device)

        for i in range(batch_size):
            for j in range(batch_size):
                # Check if sample i and sample j have identical class labels
                if class_labels[i] == class_labels[j]:
                    mask[i, j] = 1.0
        positive_mask = mask

        # temperature = self.log_logit_scale.exp()
        temperature = 0.07

        # Compute the cosine similarity between label and audio features.
        # Since the features are normalized, the dot product is equivalent to cosine similarity.
        logits_per_label = torch.matmul(label_features, audio_features.t()) / temperature
        logits_per_audio = logits_per_label.t()  # symmetric for the two modalities

        # Compute exponential scores.
        exp_logits_label = torch.exp(logits_per_label)
        exp_logits_audio = torch.exp(logits_per_audio)

        # Apply the positive mask.
        positive_logits_label = exp_logits_label * positive_mask
        positive_logits_audio = exp_logits_audio * positive_mask

        # Denominators for probabilities.
        denominator_label = exp_logits_label.sum(dim=1, keepdim=True)
        denominator_audio = exp_logits_audio.sum(dim=1, keepdim=True)

        # Compute the probabilities for positive pairs in each direction.
        prob_label = positive_logits_label.sum(dim=1) / (denominator_label.squeeze() + 1e-8)
        prob_audio = positive_logits_audio.sum(dim=1) / (denominator_audio.squeeze() + 1e-8)

        # Compute the negative log-likelihood loss for each direction.
        loss_label = -torch.log(prob_label + 1e-8)
        loss_audio = -torch.log(prob_audio + 1e-8)

        # Return the average of both losses.
        total_loss = (loss_label + loss_audio).mean()
        return total_loss


