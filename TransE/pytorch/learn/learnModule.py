import torch
import torch.nn as nn


class Szt(nn.Module):
    def __init__(self, num, dim):
        super(Szt, self).__init__()
        self.num = num
        self.dim = dim
        self.emb = nn.Embedding(num, dim)
        self.my_initialize()
        # print(self.emb.weight)
        print(self.emb.weight.size())
        # print(self.emb.weight.grad)
        aver = torch.mean(self.emb.weight, 1, keepdim=False)
        print(aver)
        print(aver.size())
    def my_initialize(self):
        # nn.init.xavier_uniform_(self.emb.weight)
        nn.init.uniform_(self.emb.weight, 0, 1)


    def forward(self, *input):
        return 0


# szt = Szt(2, 1000000)
# t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
# aver = torch.mean(t, 1, keepdim=False)
# print(t.size())
# print(aver.size())
# print(aver)
x1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float, requires_grad=True)
x2 = torch.sum(torch.abs(x1), dim=1, keepdim=True)
print(x2)
print(x1/x2)
