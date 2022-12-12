import torch
import torch.nn as nn
import torch.nn.functional as f
import time
import copy

a = list(range(1, 6))
b = list(range(6, 11))
print(a, b)


def test1():
    a = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.float)
    norm = torch.norm(a, dim=-1, keepdim=True)
    print(norm)


# for i in range(10):
#     print(i)
def test2():
    v = [1, 2, 3]
    a, b, c = tuple(v)
    print(a, b, c)
    d = 4
    print('{},{},{},{}'.format(*tuple(v), d))


def test3():
    v = torch.tensor([a, b], dtype=torch.float)
    v2 = f.normalize(v, p=2, dim=-1)
    print(v2)


def compare_velocity_tensor_list():
    total = 100
    num = 3
    v1 = list(range(3))
    start = time.time()
    ans = []
    for i in range(total):
        curr = []
        for j in range(num):
            curr.append(v1)
        ans.append(torch.tensor([v1] + curr).view(-1, 3))
    print('tensor_time:{:.8f}'.format(time.time() - start))  # 0.00073910

    start = time.time()
    ans = []
    for i in range(total):
        curr = []
        for j in range(num):
            curr.append(copy.deepcopy(v1))
        ans.append(torch.tensor([v1] + curr).view(-1, 3))
    print('tensor_copy_time:{:.8f}'.format(time.time() - start))  # 0.00130558

    start = time.time()
    ans = []
    for i in range(total):
        curr = []
        for j in range(num):
            curr.append(v1)
        ans.append([v1] + curr)
    print('list_time:{:.8f}'.format(time.time() - start))  # 0.00008464

    start = time.time()
    ans = []
    for i in range(total):
        curr = []
        for j in range(num):
            curr.append(copy.deepcopy(v1))
        ans.append([v1] + curr)
    print('list_copy_time:{:.8f}'.format(time.time() - start))  # 0.00086808

    start = time.time()
    ans = []
    h, r, t = v1[0], v1[1], v1[2]
    for i in range(total):
        curr = [[h, r, x] for x in range(num)]
        ans.append([v1] + curr)
    print('lan_time:{:.8f}'.format(time.time() - start))  # 0.00006771


def torch_ones_vs_torch_tile():
    num = 1000000
    # 两个时间差不多

    start = time.time()
    tail = torch.tensor(range(num))
    head = torch.ones_like(tail) * 100
    relation = torch.ones_like(tail) * 100
    # print(relation)
    print('torch_ones_time:{:.8f}'.format(time.time() - start))  #

    start = time.time()
    tail = torch.tensor(range(num))
    head = torch.tensor(100).tile((num,))
    relation = torch.tensor(100).tile((num,))
    # print(relation)
    print('torch_tile_time:{:.8f}'.format(time.time() - start))  #
    pass


# compare_velocity_tensor_list()
# torch_ones_vs_torch_tile()
resume = True
for i in range(10):
    print(i)
    if resume:
        i = 5
        resume = False
