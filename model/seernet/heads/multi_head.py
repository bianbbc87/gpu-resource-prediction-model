# SeerNet Multi용 metric별 독립 MLP head

import torch.nn as nn
from typing import Dict


class MultiHead(nn.Module):
    """
    Multi-metric prediction heads
    Each metric has its own MLPHead-like subnetwork
    """

    def __init__(
        self,
        input_dim,
        metrics: Dict[str, int],
        hidden_dim=256,
    ):
        """
        Args:
            input_dim: dimension of graph embedding u'
            metrics: dict {metric_name: output_dim}
                     e.g. {
                       "time": 1,
                       "sm_util": 1,
                       "mem_util": 1
                     }
        """
        super().__init__()
        self.heads = nn.ModuleDict()

        for name, out_dim in metrics.items():
            self.heads[name] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, u):
        """
        Args:
            u: graph embedding u' [F_u']

        Returns:
            predictions: dict {metric_name: Tensor}
        """
        return {
            name: head(u)
            for name, head in self.heads.items()
        }
