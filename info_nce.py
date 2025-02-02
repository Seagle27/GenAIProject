import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def InfoNCE(audio_features, label_features, class_labels):
    # Normalize embeddings
    audio_features = F.normalize(audio_features, dim=-1)
    label_features = F.normalize(label_features, dim=-1)
    logit_scale = nn.Parameter(torch.ones([]) * 1)

    # Create a mask for same-class positives
    batch_size = len(class_labels)
    mask = torch.zeros((batch_size, batch_size), dtype=torch.float32)

    for i in range(batch_size):
        for j in range(batch_size):
            # Check if sample i and sample j have identical class labels
            if class_labels[i] == class_labels[j]:
                mask[i, j] = 1.0
    positive_mask = mask

    cdist_per_image = torch.cdist(label_features, audio_features, p=2) * logit_scale
    exp_logits_img = torch.exp(cdist_per_image)
    # cdist_per_image = (torch.diag(torch.matmul(image_features,image_features.transpose(0,1)),0)+torch.diag(torch.matmul(audio_features,audio_features.transpose(0,1)),0)-2*torch.matmul(image_features,audio_features.transpose(0,1)))**(1/2)
    cdist_per_aud = cdist_per_image.t()
    exp_logits_aud = torch.exp(cdist_per_aud)
    positive_logits_img = exp_logits_img * positive_mask;
    positive_logits_aud = exp_logits_aud * positive_mask;
    denominator_img = exp_logits_img.sum(dim=1, keepdim=True)
    denominator_aud = exp_logits_aud.sum(dim=1, keepdim=True)

    loss1 = positive_logits_img.sum(dim=1) / denominator_img.squeeze()
    loss2 = positive_logits_aud.sum(dim=1) / denominator_aud.squeeze()
    total_loss = -torch.log(loss1 + loss2 + 1e-8).mean()
    # total_loss = (self.loss_img(-cdist_per_image, ground_truth) + self.loss_aud(-cdist_per_aud, ground_truth)) / 2

    return total_loss