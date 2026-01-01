# ρ_v→z

import torch

def softmax_aggregation(node_features):
    """
    Softmax aggregation over nodes.

    Args:
        node_features: Tensor [N_v, F_v']

    Returns:
        aggregated: Tensor [F_v']
    """
    # 논문 구현: node importance 기반 softmax
    scores = node_features.mean(dim=1)
    weights = torch.softmax(scores, dim=0).unsqueeze(-1)
    return (weights * node_features).sum(dim=0)
