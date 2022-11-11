import torch
import torch.nn as nn
import random
from data_process import logger


def norm2(emb):
    """
    :param emb: nn.Embedding
    :return:
    """
    n_emb = emb.weight.data.clone()  # clone + data(不在计算图中)
    n_emb = n_emb / torch.sum(torch.pow(n_emb, 2), dim=1, keepdim=True)  # 2范数 忽略开方
    emb.weight.data.copy_(n_emb)


def unique_digit(x, num):
    n_x = random.randint(0, num - 1)
    loop = 0
    while n_x == x:
        n_x = random.randint(0, num - 1)
        loop = loop + 1
        if loop >= 100:
            logger.info('循环随机数超过100轮了 暂停')
            return x  # TODO: 不知道怎么处理 先返回原值吧
    return n_x


def norm1(emb):
    n_emb = emb.weight.data.clone()  # clone + data(不在计算图中)
    n_emb = n_emb / torch.sum(torch.abs(n_emb), dim=1, keepdim=True)
    emb.weight.data.copy_(n_emb)


class TransE(nn.Module):
    def __init__(self, id_triples, id2e, id2r, margin, dim, norm=2):
        super(TransE, self).__init__()
        self.id_triples = id_triples
        self.id2e = id2e
        self.id2r = id2r
        self.margin = margin
        self.dim = dim
        self.norm = norm

        self.e_len = len(id2e)
        self.r_len = len(id2r)
        self.triple_len = len(id_triples)

        self.e_emb = nn.Embedding(self.e_len, dim)
        self.r_emb = nn.Embedding(self.r_len, dim)

    def my_initialize(self):
        nn.init.xavier_uniform_(self.r_emb.weight)
        norm2(self.r_emb)  # 规范化 除以自身的2范数
        nn.init.xavier_uniform_(self.e_emb.weight)

    def norm2_entity(self):
        norm2(self.e_emb)

    def create_neg_triples(self, pos_triples):
        neg_h, neg_r, neg_t = [], [], []
        triple_num = len(pos_triples[0])
        for i in range(triple_num):
            h = pos_triples[0][i]
            r = pos_triples[1][i]
            t = pos_triples[2][i]

            p = random.random()
            if p < 0.5:
                h = unique_digit(h, self.e_len)
            else:
                t = unique_digit(t, self.e_len)

            neg_h.append(h)
            neg_r.append(r)
            neg_t.append(t)

        neg_triples = []
        for neg in [neg_h, neg_r, neg_t]:
            neg_triples.append(torch.tensor(neg))
        return neg_triples

    def forward(self, pos_triple, neg_triple):

        pass
