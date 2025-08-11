import torch
import torch.nn.functional as F

def batch_hard_triplet_loss(labels, embeddings, margin, device):
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

    mask_positive = (labels.unsqueeze(1) == labels.unsqueeze(0)).bool().to(device)
    mask_negative = ~mask_positive
    
    anchor_positive_dist = pairwise_dist.clone()
    anchor_positive_dist[~mask_positive] = -float('inf')
    hardest_positive_dist, _ = torch.max(anchor_positive_dist, dim=1)

    anchor_negative_dist = pairwise_dist.clone()
    anchor_negative_dist[mask_positive] = float('inf')
    hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1)
    
    triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + margin)
    
    return triplet_loss.mean()