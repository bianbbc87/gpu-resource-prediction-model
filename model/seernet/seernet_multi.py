# SeerNet-Multi + PCGrad

import torch
import torch.nn as nn
from typing import Dict

from model.seernet.blocks import SeerBlock
from model.seernet.heads import MultiHead

class PCGrad:
    """
    Project Conflicting Gradients
    (Yu et al., NeurIPS 2020)
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, losses):
        """
        Args:
            losses: list of scalar losses (one per task)
        """
        grads = []
        params = [
            p for group in self.optimizer.param_groups
            for p in group["params"] if p.requires_grad
        ]

        for loss in losses:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grads.append([
                p.grad.clone() if p.grad is not None else None
                for p in params
            ])

        # project conflicting gradients
        final_grads = grads[0]
        for i in range(1, len(grads)):
            for j, (g, g_i) in enumerate(zip(final_grads, grads[i])):
                if g is None or g_i is None:
                    continue
                dot = torch.dot(g.flatten(), g_i.flatten())
                if dot < 0:
                    g -= dot / (g_i.norm() ** 2 + 1e-6) * g_i

        # apply gradients
        self.optimizer.zero_grad()
        for p, g in zip(params, final_grads):
            if g is not None:
                p.grad = g
        self.optimizer.step()

class SeerNetMulti(nn.Module):
    """
    SeerNet-Multi: multi-metric predictor
    """

    def __init__(
        self,
        node_dim,
        edge_dim,
        global_dim,
        metrics: Dict[str, int],
        hidden_dim=256,
        global_node_dim=256,
    ):
        super().__init__()

        self.seer_block = SeerBlock(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            global_node_dim=global_node_dim,
        )

        self.z_init = nn.Parameter(
            torch.zeros(global_node_dim)
        )

        self.heads = MultiHead(
            input_dim=hidden_dim,
            metrics=metrics,
            hidden_dim=hidden_dim,
        )

    def forward(self, perfgraph):
        """
        Returns:
            dict {metric_name: prediction}
        """
        V = perfgraph.V
        E = perfgraph.E
        edge_index = perfgraph.edge_index
        u = perfgraph.u

        z = self.z_init

        _, u_prime, _ = self.seer_block(
            V=V,
            E=E,
            edge_index=edge_index,
            u=u,
            z=z,
        )

        return self.heads(u_prime)