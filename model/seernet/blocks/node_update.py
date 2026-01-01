# φ_v

import torch
import torch.nn as nn


class NodeUpdate(nn.Module):
    """
    φ_v(ē'_i, v_i, z, u)
    """

    def __init__(self, edge_hidden_dim, node_dim, global_node_dim, global_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_hidden_dim + node_dim + global_node_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, edge_agg, V, z, u):
        """
        edge_agg: [N_v, F_e']
        V: [N_v, F_v]
        z: [F_z]
        u: [F_u]
        """
        N = V.size(0)
        z_expand = z.unsqueeze(0).expand(N, -1)
        u_expand = u.unsqueeze(0).expand(N, -1)
        x = torch.cat([edge_agg, V + z_expand, u_expand], dim=-1)
        return self.mlp(x)
