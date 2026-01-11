# z, GNPB

import torch
import torch.nn as nn


class GlobalNode(nn.Module):
    """
    GNPB: global node z
    φ_z( v̄'_z , z )
    """

    def __init__(self, node_hidden_dim, global_node_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_hidden_dim, global_node_dim),
            nn.LayerNorm(global_node_dim),  # 안정적 학습 지원: 수치가 널뛰는 현상을 막아 학습 성능을 일정하게 유지
            nn.ReLU(),
            nn.Linear(global_node_dim, global_node_dim),
        )

    def forward(self, V_prime, z):
        """
        V_prime: [N_v, F_v']
        z: [F_z]
        """
        # ρ_v→z : softmax aggregation
        weights = torch.softmax(V_prime.mean(dim=1), dim=0).unsqueeze(-1)
        v_bar = (weights * V_prime).sum(dim=0)
        return self.mlp(v_bar + z)
