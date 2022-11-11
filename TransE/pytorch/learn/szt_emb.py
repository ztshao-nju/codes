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

# emb = nn.Embedding(1, 2)
# print(emb.weight, type(emb.weight))
# print(emb.weight.requires_grad)
# print(emb.weight.data, type(emb.weight.data))
# print(emb.weight.data.requires_grad)
test_weight = False
if test_weight:
    pre_emb = torch.tensor([[1, 2]], dtype=torch.float, requires_grad=True)
    emb = nn.Embedding.from_pretrained(pre_emb)
    print(emb.weight.requires_grad, emb.weight.data.requires_grad)
    # False False 两个都不能求梯度
    emb.weight.requires_grad = True
    emb.weight.data.requires_grad = True
    print(emb.weight, type(emb.weight))
    print(emb.weight.requires_grad)
    print(emb.weight.data, type(emb.weight.data))
    print(emb.weight.data.requires_grad)  # 仍然是False

    print('---------------------------------------------------')
    y = torch.pow(emb.weight, 2).sum()
    y.backward()
    print('emb的梯度是', emb.weight.grad)  # 梯度是2x

    print('---------------------------------------------------')
    # fs = emb.weight.data.clone().pow(2)
    # fs.requires_grad = True
    # print(fs)
    # emb.weight.grad.zero_()
    # print(emb.weight.grad)
    # y = (torch.pow(fs, 2) + torch.pow(emb.weight, 2)).sum()
    # y.backward()
    # print('再次输出梯度')
    # print(emb.weight.grad)
    # print(fs.grad)


pre_emb = torch.tensor([[1, 2], [3, 4]], dtype=torch.float, requires_grad=True)
emb = nn.Embedding.from_pretrained(pre_emb)
emb.weight.requires_grad = True

# tmp = emb.weight.data.clone()
# tmp.requires_grad = True
# sum = torch.sum(torch.pow(tmp, 2), dim=1, keepdim=True)
# print(sum)  # tensor([ 5., 25.], grad_fn=<SumBackward2>)
# print(tmp.size(), sum.size())  # torch.Size([2, 2]) torch.Size([2, 1])
# tmp = tmp / sum
# print(tmp)  # 得到规范化后的数据
# emb.weight.data = tmp.clone()
# print('output emb.weight & emb.weight.data --------------')
# print(emb.weight)
# print(emb.weight.data)
# print(emb.weight.requires_grad, emb.weight.data.requires_grad)

tmp = emb.weight.data.clone()
tmp = tmp / torch.sum(torch.pow(tmp, 2), dim=1, keepdim=True)
emb.weight.data = tmp.clone()
# emb.weight.data.copy_(tmp)
print('output emb.weight & emb.weight.data --------------')
print(emb.weight)
print(emb.weight.data)
print(emb.weight.requires_grad, emb.weight.data.requires_grad)