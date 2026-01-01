# 2-layer MLP

import torch.nn as nn

class MLPHead(nn.Module):
    """
    Single-metric prediction head
    """

    def __init__(self, input_dim, hidden_dim=256, output_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, u):
        """
        Args:
            u: graph embedding u' [F_u']

        Returns:
            prediction: Tensor [output_dim]
        """
        return self.mlp(u)
