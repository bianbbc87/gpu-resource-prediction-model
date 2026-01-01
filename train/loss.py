# MSE / multi-metric

import torch.nn as nn


def build_loss(loss_type="mse"):
    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type == "l1":
        return nn.L1Loss()
    raise ValueError(f"Unknown loss type: {loss_type}")
