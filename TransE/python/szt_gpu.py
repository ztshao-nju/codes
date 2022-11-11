import torch
from torch import nn

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
print(torch.device('cpu'))
print(torch.cuda.device('cuda'))
print(torch.cuda.device('cuda:1'))
print(torch.cuda.device_count())  # 查看GPU数量

