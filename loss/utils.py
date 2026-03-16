import torch
import torch.nn.functional as F


def distance(x, dist_type, normalize_input=True):
    """
    Args:
        x: embedding tensor
        dist_type: distance metric type
            "cosine" means the negative cosine similarity
            ...
        normalize_input: l2-normalize x or not
    Returns:
        distance matrix, which contains the distance between any two embeddings
    """
    if normalize_input:
        x = F.normalize(x, p=2, dim=1)
    if dist_type == "cosine":
        return -(x @ x.T)
    elif dist_type == "euclidean":
        return torch.cdist(x, x, p=2)
    elif dist_type == "squared_euclidean":
        return ((x.unsqueeze(1) - x.unsqueeze(0)) ** 2).sum(-1)
    else:
        raise NotImplementedError(f"not support: {dist_type}")
