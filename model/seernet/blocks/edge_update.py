# φ_e

import torch
import torch.nn as nn


class EdgeUpdate(nn.Module):
    """
    φ_e(e_j, v_sj, v_tj)
    """

    def __init__(self, edge_dim, node_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 수치 안정성 확보: 층 통과 시 결과값이 널뛰지 않게 평균/분산을 조정해주는 안전 밸브
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, E, V, edge_index):
        """
        E: [N_e, F_e]
        V: [N_v, F_v]
        edge_index: [2, N_e]
        """
        src, dst = edge_index
        v_src = V[src]
        v_dst = V[dst]
        x = torch.cat([E, v_src, v_dst], dim=-1)
        return self.mlp(x)
