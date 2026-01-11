# φ_u

import torch
import torch.nn as nn


class GlobalUpdate(nn.Module):
    """
    φ_u( v̄'_u , z' , u )
    """

    def __init__(self, node_hidden_dim, global_node_dim, global_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_hidden_dim + global_node_dim + global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 수치 밸런스 조정: 폭발적인 그래디언트 현상을 방지하여 학습 중단을 차단
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, v_bar_u, z_prime, u):
        x = torch.cat([v_bar_u, z_prime, u], dim=-1)
        return self.mlp(x)
