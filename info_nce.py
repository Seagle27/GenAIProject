import torch
import torch.nn.functional as F


def multi_positive_info_nce_loss(audio_emb, label_emb, class_labels, temperature=0.07):
    # Normalize embeddings
    audio_emb = F.normalize(audio_emb, dim=-1)
    label_emb = F.normalize(label_emb, dim=-1)

    # Compute similarity matrix
    logits = torch.matmul(audio_emb, label_emb.T) / temperature

    # Create a mask for same-class positives
    positive_mask = create_positive_mask_exact(class_labels).to(audio_emb.device)

    # Compute probabilities
    exp_logits = torch.exp(logits)
    positive_logits = exp_logits * positive_mask
    denominator = exp_logits.sum(dim=1, keepdim=True)

    # Compute loss: -log(sum(positive_logits) / sum(exp_logits))
    positive_probs = positive_logits.sum(dim=1) / denominator.squeeze()
    loss = -torch.log(positive_probs + 1e-8).mean()  # Add epsilon for numerical stability
    return loss


def create_positive_mask_exact(class_labels):
    """
    Create a pairwise positive mask based on identical class labels.

    Args:
        class_labels (list of list): A list where each sublist contains the class labels for a sample.

    Returns:
        torch.Tensor: A mask of shape (B, B) where mask[i, j] = 1 if class_labels[i] == class_labels[j].
    """
    batch_size = len(class_labels)
    mask = torch.zeros((batch_size, batch_size), dtype=torch.float32)

    for i in range(batch_size):
        for j in range(batch_size):
            # Check if sample i and sample j have identical class labels
            if class_labels[i] == class_labels[j]:
                mask[i, j] = 1.0

    return mask

