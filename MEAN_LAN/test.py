import torch

a = torch.tensor([[1, 2, 3, 4, 5],[6, 7, 8, 9, 10]], dtype=torch.float)
norm = torch.norm(a, dim=-1, keepdim=True)
print(norm)
a2 = a / norm


for i in range(10):
    print(i)