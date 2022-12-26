import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy

device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
    v2 = F.normalize(v, p=2, dim=-1)
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

def test_grad():
    x1 = torch.tensor([1, 2], dtype=torch.float, requires_grad=True)
    x2 = torch.tensor([3, 4], dtype=torch.float, requires_grad=True)
    x3 = torch.tensor([5, 6], dtype=torch.float, requires_grad=True)
    # y = (torch.pow(x1, 3) + torch.pow(x2, 2) + x3).sum()
    # y.backward()
    # x4 = x2.clone()
    # print(x1.grad, x2.grad, x3.grad, x4.grad, x4.requires_grad)
    # 梯度分别是 3x^2, 2x, 1, None
    # tensor([ 3., 12.]) tensor([6., 8.]) tensor([1., 1.]) None True
    t_id_1 = torch.tensor(list(range(5)))
    emb = nn.Embedding(10, 10)
    # emb.requires_grad = False
    t_id = t_id_1 + 1
    print(t_id.grad)
    t_emb = emb(t_id)
    print(t_emb.grad)


def test_mask():
    batch_size = 2
    max_neighbor = 3
    dim = 3
    cnt_e = 10
    batch_nei_e_Tr_emb = torch.tensor([
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
        [[0, 1, 2],
         [3, 4, 5],
         [6, 7, 8]]
    ])  # (batch_size, max_neighbor, dim)
    batch_nei_rid = torch.tensor([
        [1, 10, 6],
        [4, 8, 10]
    ])  # (batch_size, max_neighbor)
    mask_emb = torch.cat([torch.ones([cnt_e, 1]), torch.zeros([1, 1])], dim=0)  # (cnt_e+1, 1)
    mask = mask_emb[batch_nei_rid]
    print(mask)
    ans = batch_nei_e_Tr_emb * mask
    print(ans)

def test_detach():
    ans = [0, 0]
    rank = torch.tensor([[1], [5]])
    hits_nums = [1, 3]
    for _index, hits in enumerate(hits_nums):
        ans[_index] += torch.sum(rank <= hits).detach_()

    print(ans)

def test_differentiable():
    v = torch.tensor(list(range(-5, 3)), dtype=torch.float16, requires_grad=True, device=device)
    print(v.requires_grad)
    v2 = torch.tensor(list(range(2, 10)), dtype=torch.float16, requires_grad=True, device=device)
    ans_abs = torch.abs(v)
    ans_sum = torch.sum(v)
    ans_f = F.relu(v)
    # ans_max = torch.max(v, 0)
    print('torch.abs: {}'.format(ans_abs.grad_fn))
    print('torch.sum: {}'.format(ans_sum.grad_fn))
    print('F.relu', ans_f.grad_fn)
    # print('torch.max: {}'.format(ans_max.grad_fn))


def test_logger():
    from options.logger import logFrame
    log1 = logFrame()
    logger1 = log1.getlogger('file1')
    log2 = logFrame()
    logger2 = log2.getlogger('file2')
    logger1.info('test')
    logger2.info('test2')

def test_loss():
    neg = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.float)
    pos = torch.tensor([[3, 4, 5], [1, 2, 3]], dtype=torch.float)
    margin = torch.tensor([1])
    ans = margin - pos + neg
    print(ans)
    ans = torch.mean(F.relu(ans), dim=-1)
    print(ans)

test_loss()
# test_logger()
# test_differentiable()
# test_detach()
# test_mask()
# test_grad()
# compare_velocity_tensor_list()
# torch_ones_vs_torch_tile()


