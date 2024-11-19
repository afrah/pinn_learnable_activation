import torch


def MSE(true, pred):
    return torch.sum(torch.mean((true - pred) ** 2, dim=0))
