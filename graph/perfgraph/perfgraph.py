# PerfGraph 데이터 구조

from dataclasses import dataclass
import torch

@dataclass
class PerfGraph:
    """
    PerfGraph G = (u, V, E)
    """

    # global feature
    u: torch.Tensor          # [F_u]

    # node features
    V: torch.Tensor          # [N_v, F_v]

    # edge features
    E: torch.Tensor          # [N_e, F_e]

    # topology
    edge_index: torch.Tensor # [2, N_e]

    def to_pyg(self):
        """
        Optional: PyTorch Geometric 변환
        """
        from torch_geometric.data import Data

        return Data(
            x=self.V,
            edge_index=self.edge_index,
            edge_attr=self.E,
            u=self.u,
        )
