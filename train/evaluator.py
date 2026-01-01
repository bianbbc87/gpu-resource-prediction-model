# MAPE, RMSPE 계산

import torch
from .metrics import mape, rmspe


class Evaluator:
    """
    Evaluation utilities
    """

    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()

        mape_list = []
        rmspe_list = []

        for perfgraph, label in dataloader:
            perfgraph.V = perfgraph.V.to(self.device)
            perfgraph.E = perfgraph.E.to(self.device)
            perfgraph.edge_index = perfgraph.edge_index.to(self.device)
            perfgraph.u = perfgraph.u.to(self.device)

            pred = self.model(perfgraph).squeeze()

            target = torch.tensor(
                float(label["infer"].split("|")[0]),
                device=self.device,
            )

            mape_list.append(mape(pred, target).item())
            rmspe_list.append(rmspe(pred, target).item())

        return {
            "MAPE": sum(mape_list) / len(mape_list),
            "RMSPE": sum(rmspe_list) / len(rmspe_list),
        }
