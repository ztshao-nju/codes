import torch
import torch.nn as nn

class Framework(nn.Module):

    def __init__(self, g, args, device):
        ##############################################
        super().__init__()
        self.train_g = torch.tensor(g.train_g, dtype=torch.long).to(device)  # (e_num, max_neighbor, 2)
        self.train_w = torch.tensor(g.train_w, dtype=torch.float).to(device)  # (e_num, max_neighbor)
        self.corr = torch.tensor(g.corr, dtype=torch.float).to(device)   # self.corr[j][i] 表示 P(i->j)

        self.cnt_e = g.cnt_e
        self.cnt_r = g.cnt_r

        self.aggregate_type = args.aggregate_type
        self.use_logic_attention = args.use_logic_attention
        self.use_nn_attention = args.use_nn_attention

        self.max_neighbor = args.max_neighbor
        self.dim = args.dim

        self.margin = args.margin
        self.weight_decay = args.weight_decay
        self.learning_rate = args.learning_rate

        self.loss_function = args.loss_function
        self.score_function = args.score_function

        ##############################################
        self.e_emb = nn.Embedding(self.cnt_e, self.dim)
        self.r_emb = nn.Embedding(self.cnt_r, self.dim)
        if self.loss_function == "margin":
            self.loss_function = nn.MarginRankingLoss(self.margin, reduction='mean')

        from .encoder import Encoder_ATTENTION
        self.encoder = Encoder_ATTENTION(self.cnt_e, self.cnt_r, self.dim)

    # 计算经过 aggregate 之后的e_i^O
    def calcu_emb_o(self, batch_neighbor_rid, batch_neighbor_eid):
        batch_neighbor_emb = self.e_emb(batch_neighbor_eid)
        pass


    # 1. task 1
    def fun1(self, batch_e_id, batch_q_rid):
        """
        :param batch_e_id: (batch_size, )
        :param batch_q_rid: (batch_size, )
        :return:
        """
        # 1. 找每个 e_id 的 neighbor 信息: nei_rid, nei_eid, nei_rw: (batch_size, max_neighbor)
        #    找到每个 e_id 的 neighobbr 的 nei_eid 的 emb: e_j^I   :(batch_size, max_neighbor, dim)
        #    找到每个batch的q_rid    batch_q_rid     :(batch_size, )
        #    找到每个batch的neighbor的r 和 q 的 corr    :(batch_size, max_neighbor)
        batch_nei_rid = self.train_g[batch_e_id, :, 0]  # (batch_size, max_neighbor)
        batch_nei_eid = self.train_g[batch_e_id, :, 1]  # (batch_size, max_neighbor)
        batch_nei_e_emb = self.e_emb(batch_nei_eid)  # (batch_size, max_neighbor, dim)
        batch_nei_r_denominator = self.train_w[batch_e_id, :]  # (batch_size, max_neighbor)   分母

        batch_nei_r_numerator = self.corr[batch_nei_rid]  # (batch_size, max_neighbor)   分子
        batch_nei_rw = batch_nei_r_numerator / batch_nei_r_denominator
        self.encoder(batch_nei_rid, batch_nei_e_emb, batch_nei_rw, batch_q_rid)

        pass

    # 2. task 2
    def task2(self, batch_pos_triplet_emb, batch_neg_triplet_emb):
        score_pos = self.get_score(batch_pos_triplet_emb)
        score_neg = self.get_score(batch_neg_triplet_emb)
        pass

    # TODO: 提供 pos/neg 数据 返回loss1+loss2
    def forward(self, batch_pos_triplet_id, batch_neg_triplet_id):
        hp, rp, tp = batch_pos_triplet_id
        self.fun1(hp, rp)
        # (tp, rp+cnt_r)
        pass

    def get_score(self, batch_triplet):
        h, r, t = batch_triplet
        if self.score_function == 'TransE':
            from .score_function import transe
            return transe(h, r, t)
        pass