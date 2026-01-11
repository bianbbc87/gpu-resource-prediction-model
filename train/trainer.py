# train loop

import torch


class Trainer:
    """
    Training loop for SeerNet / SeerNetMulti
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device="cuda",
        pcgrad=None,
        logger=None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.pcgrad = pcgrad
        self.logger = logger

    def train_step(self, perfgraph, label):
        self.model.train()

        # move tensors
        perfgraph.V = perfgraph.V.to(self.device)
        perfgraph.E = perfgraph.E.to(self.device)
        perfgraph.edge_index = perfgraph.edge_index.to(self.device)
        perfgraph.u = perfgraph.u.to(self.device)

        output = self.model(perfgraph)

        # multi-metric
        if isinstance(output, dict):
            losses = []
            for k, pred in output.items():
                target = label.to(self.device)  # Use normalized label
                losses.append(self.loss_fn(pred.squeeze(), target))

            if self.pcgrad:
                self.pcgrad.step(losses)
                return sum(l.item() for l in losses)

            loss = sum(losses)
        else:
            # label is already normalized tensor from dataset
            target = label.to(self.device)
            loss = self.loss_fn(output.squeeze(), target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
