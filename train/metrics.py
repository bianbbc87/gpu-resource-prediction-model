import torch


def mape(pred, target, eps=1e-6):
    return torch.mean(torch.abs((pred - target) / (target + eps)))


def rmspe(pred, target, eps=1e-6):
    return torch.sqrt(
        torch.mean(((pred - target) / (target + eps)) ** 2)
    )
