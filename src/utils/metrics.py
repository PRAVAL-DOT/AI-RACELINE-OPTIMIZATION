import torch
import numpy as np
from typing import Dict

@torch.no_grad()
def evaluate_embeddings_topk(embeddings: np.ndarray, labels: np.ndarray, topk=(1, 5)) -> Dict[str, float]:
    """
    Computes k-NN classification accuracy metrics directly in the latent embedding space.
    """
    emb_tensor = torch.from_numpy(embeddings)
    lbl_tensor = torch.from_numpy(labels)
    
    # Calculate pairwise cosine distances
    norm_emb = torch.nn.functional.normalize(emb_tensor, p=2, dim=1)
    sim_matrix = torch.matmul(norm_emb, norm_emb.T)
    
    # Exclude self-matching values along diagonal components
    sim_matrix.fill_diagonal_(-1.0)
    
    max_k = max(topk)
    _, topk_indices = torch.topk(sim_matrix, k=max_k, dim=1)
    
    ret_metrics = {}
    for k in topk:
        correct = 0
        for i in range(len(labels)):
            neighbor_labels = lbl_tensor[topk_indices[i, :k]]
            if lbl_tensor[i] in neighbor_labels:
                correct += 1
        ret_metrics[f"Embedding_Top-{k}_Acc"] = (correct / len(labels)) * 100
        
    return ret_metrics