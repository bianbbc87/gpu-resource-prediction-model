# ρ_e→v

import torch

def mean_aggregation(edge_features, edge_index, num_nodes):
    """
    Mean aggregation for edges to nodes.

    Args:
        edge_features: Tensor [N_e, F_e']
        edge_index: Tensor [2, N_e]
        num_nodes: int

    Returns:
        aggregated: Tensor [N_v, F_e']
    """
    dst = edge_index[1]
    out = torch.zeros(
        num_nodes,
        edge_features.size(1),
        device=edge_features.device,
    )

    out.index_add_(0, dst, edge_features)

    deg = torch.bincount(dst, minlength=num_nodes).clamp(min=1)
    out = out / deg.unsqueeze(-1)

    return out
