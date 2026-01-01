# SeerNet = SeerBlock + Head 

import torch
import torch.nn as nn

from model.seernet.blocks import SeerBlock
from model.seernet.heads import MLPHead


class SeerNet(nn.Module):
    """
    SeerNet: single-metric performance predictor
    """

    def __init__(
        self,
        node_dim,
        edge_dim,
        global_dim,
        hidden_dim=256,
        global_node_dim=256,
        output_dim=1,
    ):
        super().__init__()

        # SeerBlock
        self.seer_block = SeerBlock(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            global_node_dim=global_node_dim,
        )

        # initial global node z
        self.z_init = nn.Parameter(
            torch.zeros(global_node_dim)
        )

        # prediction head
        self.head = MLPHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

    def forward(self, perfgraph):
        """
        Args:
            perfgraph: PerfGraph
              - perfgraph.V : [N_v, F_v]
              - perfgraph.E : [N_e, F_e]
              - perfgraph.edge_index : [2, N_e]
              - perfgraph.u : [F_u]

        Returns:
            prediction: Tensor [output_dim]
        """

        V = perfgraph.V
        E = perfgraph.E
        edge_index = perfgraph.edge_index
        u = perfgraph.u

        # global node z
        z = self.z_init

        # SeerBlock
        V_prime, u_prime, z_prime = self.seer_block(
            V=V,
            E=E,
            edge_index=edge_index,
            u=u,
            z=z,
        )

        # prediction
        out = self.head(u_prime)
        return out
