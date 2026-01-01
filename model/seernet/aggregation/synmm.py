# SynMM (Figure 3)

import torch
import torch.nn as nn

class SynMM(nn.Module):
    """
    Synergistic Max-Mean Aggregation
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.linear = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, node_features):
        """
        Args:
            node_features: Tensor [N_v, F_v']

        Returns:
            aggregated: Tensor [F_v']
        """
        v_mean = node_features.mean(dim=0)
        v_max = node_features.max(dim=0).values
        x = torch.cat([v_mean, v_max], dim=-1)
        return self.linear(x)


def synmm_aggregation(node_features, synmm_layer: SynMM):
    """
    Functional wrapper (optional)

    Args:
        node_features: Tensor [N_v, F_v']
        synmm_layer: SynMM module

    Returns:
        aggregated: Tensor [F_v']
    """
    return synmm_layer(node_features)
