# SeerBlock 전체

import torch
import torch.nn as nn

from .edge_update import EdgeUpdate
from .node_update import NodeUpdate
from .global_node import GlobalNode
from .global_update import GlobalUpdate


class SeerBlock(nn.Module):
    """
    SeerBlock (논문 Figure 2)
    """

    def __init__(
        self,
        node_dim,
        edge_dim,
        global_dim,
        hidden_dim=256,
        global_node_dim=256,
    ):
        super().__init__()

        self.edge_update = EdgeUpdate(edge_dim, node_dim, hidden_dim)

        self.node_update = NodeUpdate(
            edge_hidden_dim=hidden_dim,
            node_dim=node_dim,
            global_node_dim=global_node_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
        )

        self.global_node = GlobalNode(
            node_hidden_dim=hidden_dim,
            global_node_dim=global_node_dim,
        )

        self.global_update = GlobalUpdate(
            node_hidden_dim=2 * hidden_dim,  # mean + max concat
            global_node_dim=global_node_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, V, E, edge_index, u, z):
        """
        Inputs:
            V: [N_v, F_v]
            E: [N_e, F_e]
            edge_index: [2, N_e]
            u: [F_u]
            z: [F_z]
        Outputs:
            V', u', z'
        """

        # φ_e
        E_prime = self.edge_update(E, V, edge_index)

        # ρ_e→v (mean aggregation)
        N = V.size(0)
        edge_agg = torch.zeros(N, E_prime.size(1), device=V.device)
        dst = edge_index[1]
        edge_agg.index_add_(0, dst, E_prime)
        deg = torch.bincount(dst, minlength=N).clamp(min=1).unsqueeze(-1)
        edge_agg = edge_agg / deg

        # φ_v
        V_prime = self.node_update(edge_agg, V, z, u)

        # φ_z
        z_prime = self.global_node(V_prime, z)

        # ρ_v→u (SynMM: mean + max → concat)
        v_mean = V_prime.mean(dim=0)
        v_max = V_prime.max(dim=0).values
        v_bar_u = torch.cat([v_mean, v_max], dim=-1)

        # φ_u
        u_prime = self.global_update(v_bar_u, z_prime, u)

        return V_prime, u_prime, z_prime
