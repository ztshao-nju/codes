import torch
import torch.nn as nn
import random
from data_process import logger


def get_norm1(vec):
    return torch.sum(torch.abs(vec), dim=1)


def get_norm2(vec):
    return torch.sum(torch.pow(vec, 2), dim=1)  # in order to save time, remove sqrt
    # return torch.sqrt(torch.sum(torch.pow(vec, 2), dim=1))


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
    def __init__(self, id_triples, id2e, id2r, margin, dim, device, norm=2):
        super(TransE, self).__init__()
        self.id_triples = id_triples
        self.id2e = id2e
        self.id2r = id2r
        self.margin = margin
        self.dim = dim
        self.device = device
        self.norm = norm

        self.e_len = len(id2e)
        self.r_len = len(id2r)
        self.triple_len = len(id_triples)

        self.e_emb = nn.Embedding(self.e_len, dim)
        self.r_emb = nn.Embedding(self.r_len, dim)
        self.criterion = nn.MarginRankingLoss(self.margin, reduction='mean')

    def my_initialize(self):
        nn.init.xavier_uniform_(self.r_emb.weight)
        norm2(self.r_emb)  # 规范化 除以自身的2范数
        nn.init.xavier_uniform_(self.e_emb.weight)

    def norm2_entities(self):
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

    def calcu_dist(self, triple):
        h, r, t = triple
        h = h.to(self.device)
        r = r.to(self.device)
        t = t.to(self.device)
        h_emb = self.e_emb(h)
        r_emb = self.r_emb(r)
        t_emb = self.e_emb(t)
        res = h_emb + r_emb - t_emb
        return get_norm1(res) if self.norm == 1 else get_norm2(res)

    # calcu_loss
    def forward(self, pos_triple, neg_triple):
        # nn.MarginRankingLoss(margin)
        # loss(x1, x2, y) = max(0, -y * (x1 - x2) + margin)
        # TransE Loss: max(0, d - d' + margin)
        # therefore, y = -1, x1 = d, x2 = d', margin = margin

        x1 = self.calcu_dist(pos_triple)
        x2 = self.calcu_dist(neg_triple)
        y = torch.ones(1, 1)
        num = len(x1)
        y = y.new_full((1, num), -1).to(self.device)
        return self.criterion(x1, x2, y)  # x1 x2 y size=[N]
