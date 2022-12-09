import torch

def transe(h, r, t):
    return -torch.sum(torch.abs(h + r - t), dim=1)
