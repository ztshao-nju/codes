import torch
import torch.nn as nn
import torch.nn.functional as f
from .encoder import Encoder_ATTENTION, Encoder_Mean

class Framework(nn.Module):

    def __init__(self, g, args, device):
        super().__init__()
        self.device = device

        self.train_g = torch.tensor(g.train_g, dtype=torch.long).to(device)  # (e_num, max_neighbor, 2)
        self.train_w = torch.tensor(g.train_w, dtype=torch.float).to(device)  # (e_num, max_neighbor)
        self.corr = torch.tensor(g.corr, dtype=torch.float).to(device)   # self.corr[j][i] 表示 P(i->j)

        self.cnt_e = g.cnt_e
        self.cnt_r = g.cnt_r

        self.num_neg = args.num_neg
        self.aggregate_type = args.aggregate_type
        self.use_logic_attention = args.use_logic_attention
        self.use_nn_attention = args.use_nn_attention

        self.max_neighbor = args.max_neighbor
        self.dim = args.dim

        self.margin = args.margin
        self.weight_decay = args.weight_decay
        self.learning_rate = args.learning_rate

        # loss(x1, x2, y) = max(0, -y*(x1-x2)+margin)
        # max(0, -(-1) * (neg - pos) + margin)  y=-1    neg=x1, pos=x2
        self.loss_function = nn.MarginRankingLoss(margin=self.margin, reduction='sum')

        ##############################################
        self.e_emb = nn.Embedding(self.cnt_e + 1, self.dim)  # +1 是为了处理没有遇到的实体 id是cnt_e
        self.r_emb = nn.Embedding(self.cnt_r, self.dim)

        if self.aggregate_type == 'attention':
            self.encoder = Encoder_ATTENTION(self.cnt_e, self.cnt_r, self.dim)
        elif self.aggregate_type == 'mean':
            self.encoder = Encoder_Mean(self.cnt_r, self.dim)
        self.encoder.to(device)

    def encoder_eout(self, batch_e_id, batch_q_rid):
        """
        :param batch_e_id: (batch_size, )
        :param batch_q_rid: (batch_size, )
        :return:
        """
        # 1 找每个 e_id 的 neighbor 信息: nei_rid, nei_eid, nei_rw: (batch_size, max_neighbor)
        #    找到每个 e_id 的 neighobbr 的 nei_eid 的 emb: e_j^I   :(batch_size, max_neighbor, dim)
        #    找到每个batch的q_rid    batch_q_rid     :(batch_size, )
        #    找到每个batch的neighbor的r 和 q 的 corr    :(batch_size, max_neighbor)
        batch_nei_rid = self.train_g[batch_e_id, :, 0]  # (batch_size, max_neighbor)
        batch_nei_eid = self.train_g[batch_e_id, :, 1]  # (batch_size, max_neighbor)
        batch_nei_e_emb = self.e_emb(batch_nei_eid)  # (batch_size, max_neighbor, dim)

        # 2 获得 Logic attn: batch_nei_rw
        batch_nei_r_denominator = self.train_w[batch_e_id, :]  # (batch_size, max_neighbor)   分母
        # 获取每个batch里每个neighbor的 分子:     corr[qid][nei_rid]   :(batch_size, neighbor) P(r-q) = corr[q][r]
        batch_nei_q_rid = batch_q_rid.unsqueeze(-1).tile((1, self.max_neighbor))  # (batch_size, max_neighbor)
        batch_nei_r_numerator = self.corr[batch_nei_q_rid, batch_nei_rid]  # (batch_size, max_neighbor)   分子
        # batch_nei_rw = batch_nei_r_numerator / (batch_nei_r_denominator)  # Logic 权重
        batch_nei_rw = batch_nei_r_numerator / (batch_nei_r_denominator + 1)  # Logic 权重
        nan_num = torch.sum(torch.isnan(batch_nei_rw))
        # if nan_num != 0:
        #     print('nan_num:{}'.format(nan_num))
        # assert nan_num == 0

        # 3 输入encoder 获得实体的表示e_i^O

        if self.aggregate_type == 'attention':  # LAN
            e_out = self.encoder(batch_nei_rid, batch_nei_e_emb, batch_nei_rw, batch_q_rid)
        else:
            e_out = self.encoder(batch_nei_rid, batch_nei_e_emb)
        return e_out
    def decoder_score(self, h_out, t_out, r_id):
        r_out = self.r_emb(r_id)
        batch_score = self.get_score(h_out, r_out, t_out)
        return batch_score

    def task1_batch_score(self, hp, rp, tp):
        h_out = self.encoder_eout(hp, rp)
        t_out = self.encoder_eout(tp, rp + self.cnt_r)
        batch_score = self.decoder_score(h_out, t_out, rp)
        return batch_score
    def task1_loss(self, batch_pos_triplet_id, batch_neg_triplet_id):
        hp, rp, tp = batch_pos_triplet_id
        hn, rn, tn = batch_neg_triplet_id
        pos_score = self.task1_batch_score(hp, rp, tp)
        neg_score = self.task1_batch_score(hn, rn, tn).view(-1, self.num_neg).sum(dim=-1)
        y = torch.tensor(-1).tile((len(hp))).to(self.device)
        loss = self.loss_function(neg_score, pos_score, y)
        return loss

    def task2_batch_score(self, batch_triplet_id):
        h, r, t = batch_triplet_id
        hp_emb = self.e_emb(h)
        tp_emb = self.e_emb(r)
        rp_emb = self.e_emb(t)
        return self.get_score(hp_emb, tp_emb, rp_emb)
    def task2_loss(self, batch_pos_triplet_id, batch_neg_triplet_id):
        h, r, t = batch_pos_triplet_id
        pos_score = self.task2_batch_score(batch_pos_triplet_id)
        neg_score = self.task2_batch_score(batch_neg_triplet_id).view(-1, self.num_neg).sum(dim=-1)
        y = torch.tensor(-1).tile((len(h))).to(self.device)
        loss = self.loss_function(neg_score, pos_score, y)
        return loss

    def forward(self, batch_pos_triplet_id, batch_neg_triplet_id):
        loss1 = self.task1_loss(batch_pos_triplet_id, batch_neg_triplet_id)
        loss2 = self.task2_loss(batch_pos_triplet_id, batch_neg_triplet_id)
        return loss1 + loss2

    def get_score(self, h, r, t):  # 评估越好的 分数越大
        h = f.normalize(h, p=2, dim=1)
        r = f.normalize(r, p=2, dim=1)
        t = f.normalize(t, p=2, dim=1)
        return -torch.sum(torch.abs(h + r - t), dim=1)  # (batch_size, )
