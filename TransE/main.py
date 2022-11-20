import torch
import torch.nn as nn

def get_norm1(vec):
    return torch.sum(torch.abs(vec), dim=1)

def get_norm2(vec):
    # return torch.sum(torch.pow(vec, 2), dim=1)  # in order to save time, remove sqrt
    return torch.sqrt(torch.sum(torch.pow(vec, 2), dim=1))

tst = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
ans1 = get_norm1(tst)
print(ans1)
ans2 = get_norm2(tst)
print(ans2)

# emb = nn.Embedding(3, 3)
# print(emb.weight)
#
# h = torch.tensor([0, 1])
# print(emb(h))


# nn.MarginRankingLoss(margin)
# loss(x1, x2, y) = max(0, -y * (x1 - x2) + margin)
# TransE Loss: max(0, d - d' + margin)
# therefore, y = -1, x1 = d, x2 = d', margin = margin

# tensor = torch.ones((2,), dtype=torch.float64)
# tmp = tensor.new_full((3, 4), 3.141592)
# print(tmp)

# criterion = nn.MarginRankingLoss(margin=3, reduction='mean')
# x1 = torch.tensor([5, 8, 11])
# x2 = torch.tensor([8, 8, 8])
# y = torch.tensor([-1, -1, -1])
#
# # ans = criterion(x1, x2, y)
# x11 = torch.tensor([5, 0])
# x22 = torch.tensor([8, 0])
# yy = torch.ones(1, 2)
# yy = yy.new_full((1, 2), fill_value=-1)
# print('yy: ', yy)
# c_ans = criterion(x11, x22, yy)
# print(c_ans)
# # print(ans)
