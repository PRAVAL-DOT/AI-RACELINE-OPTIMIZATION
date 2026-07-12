import torch
import torch.nn as nn

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (SupCon) framework checking label assignments
    to balance relative anchor similarity indexes across batches.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        labels = labels.view(-1, 1)
        
        # Generate positive mask indicators based on driver IDs
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Calculate standard cosine matrix distances using dot products
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Numerical normalization trick for stability: subtract max value
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Eliminate self-contrast metrics from the equation
        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1, 
            torch.arange(mask.shape[0]).view(-1, 1).to(device), 
            0
        )
        mask = mask * logits_mask

        # Compute log probabilities over contrast pairs
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Calculate mean over positive elements per sequence item
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        loss = -mean_log_prob_pos.mean()
        return loss