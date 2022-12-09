import torch
import torch.nn as nn

class Encoder_Mean(nn.Module):
    def __init__(self, cnt_r, dim):
        super().__init__()
        self.w_r = nn.Embedding(cnt_r * 2 + 1, dim)

    def projection(self, e, w_r):
        norm2w_r = w_r / torch.norm(w_r, dim=-1, keepdim=True)  # 要求 w_r 2范数 为 1
        return e - torch.sum(e * norm2w_r, dim=-1, keepdim=True) * norm2w_r

    def forward(self, batch_e_emb, batch_r_id):
        """
        :param batch_e_emb: (batch_size, dim)
        :param batch_r_id:  (batch_size, 1)
        :return:
        """
        batch_wr = self.w_r(batch_r_id)  # (batch_size, dim)
        batch_e_emb = self.projection(batch_e_emb, batch_wr)  # (batch_size, dim)
        return batch_e_emb

class Encoder_ATTENTION(nn.Module):
    def __init__(self, cnt_e, cnt_r, dim):
        super().__init__()
        self.w_r = nn.Embedding(cnt_r * 2 + 1, dim)  # 抽取 w_r
        self.zq_emb = nn.Embedding(cnt_r * 2, dim)

        # 全局参数
        self.attn_W = nn.Linear(dim * 2, dim * 2)
        self.u_a = nn.Linear(dim * 2, 1)

        self.softmax = nn.Softmax(dim=-1)

    def projection(self, e, w_r):
        norm2w_r = w_r / torch.norm(w_r, dim=-1, keepdim=True)  # 要求 w_r 2范数 为 1
        return e - torch.sum(e * norm2w_r, dim=-1, keepdim=True) * norm2w_r

    def forward(self, batch_nei_rid, batch_nei_e_emb, batch_nei_rw, batch_q_rid):
        """
        每个 e_id 的 neighbor 信息: nei_rid, nei_eid, nei_rw:
        :param batch_nei_rid:        (batch_size, max_neighbor)
        :param batch_nei_e_emb:      (batch_size, max_neighbor, dim)
        :param batch_nei_rw:         (batch_size, max_neighbor) 就是Logic的注意力权重了 在framework里求完了
        :param batch_q_rid:          (batch_size, )
        :return:
        """
        triplet_num = batch_nei_rid.shape[0].value
        # 1. 计算 Tr(ej)  :(batch_size, max_neighbor, dim)
        w_r = self.w_r(batch_nei_rid)  # :(batch_size, max_neighbor, dim)
        batch_nei_e_Tr_emb = self.projection(batch_nei_e_emb, w_r)  # :(batch_size, max_neighbor, dim)

        # 2. 计算非标准化的NN注意力权重: alpha_{j|i,q}^{'}  :(batch_size, max_neighbor)
        #   2.1 获取 z_q  :(batch_size, max_neighbor, dim)
        batch_nei_q_rid = batch_q_rid.tile((1, triplet_num))
        batch_z_q = self.zq_emb(batch_nei_q_rid)  # (batch_size, max_neighbor, dim)
        #   2.2 拼贴 z_q 和 tr_e
        concat_emb = torch.cat((batch_z_q, batch_nei_e_Tr_emb), dim=-1)  # (batch_size, max_neighbor, dim * 2)
        #   2.3 经过 W_a 层
        wa_concat = self.attn_W(concat_emb)  # (batch_size, max_neighbor, dim * 2)
        #   2.4 经过 tanh层
        tanh = torch.tanh(wa_concat)  # (batch_size, max_neighbor, dim * 2)
        #   2.5 和u_a 相乘
        _alpha = self.u_a().squeeze(-1)  # (batch_size, max_neighbor)

        # 3. 计算标准化的NN注意力权重
        alpha = self.softmax(_alpha)  # (batch_size, max_neighbor)

        # 4. 计算Logic+NN权重
        attn = alpha + batch_nei_rw  # (batch_size, max_neighbor)

        pass