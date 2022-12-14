import torch
import torch.nn as nn
import torch.nn.functional as f


class Encoder_Mean(nn.Module):
    def __init__(self, no_mask, cnt_r, cnt_e, dim, device):
        super().__init__()
        self.w_r = nn.Embedding(cnt_r * 2 + 1, dim)
        self.no_mask = no_mask
        if not self.no_mask:
            self.mask_emb = torch.cat([torch.ones([cnt_e, 1]), torch.zeros([1, 1])], 0).to(device)  # (cnt_e+1, 1)

        self.init_values()
    def init_values(self):
        nn.init.xavier_normal_(self.w_r.weight)
    def projection(self, e, w_r):
        norm2w_r = f.normalize(w_r, p=2, dim=-1)
        # norm2w_r = w_r / torch.norm(w_r, dim=-1, keepdim=True)  # 要求 w_r 2范数 为 1
        return e - torch.sum(e * norm2w_r, dim=-1, keepdim=True) * norm2w_r

    def forward(self, batch_nei_rid, batch_nei_e_emb):
        w_r = self.w_r(batch_nei_rid)  # :(batch_size, max_neighbor, dim)
        batch_nei_e_Tr_emb = self.projection(batch_nei_e_emb, w_r)  # :(batch_size, max_neighbor, dim)
        # 考虑添加 mask  但是LAN代码里没加
        if not self.no_mask:
            mask = self.mask_emb[batch_nei_rid]  # (batch_size, max_neighbor, 1)
            batch_nei_e_Tr_emb = batch_nei_e_Tr_emb * mask

        return torch.mean(batch_nei_e_Tr_emb, dim=-2)  # 对邻居信息均值聚合


class Encoder_ATTENTION(nn.Module):
    def __init__(self, no_mask, cnt_e, cnt_r, dim, use_logic_attention, use_nn_attention, device):
        super().__init__()
        self.w_r = nn.Embedding(cnt_r * 2 + 1, dim)  # 抽取 w_r
        self.zq_emb = nn.Embedding(cnt_r * 2, dim)

        # 全局参数
        self.attn_W = nn.Linear(dim * 2, dim * 2)
        self.u_a = nn.Linear(dim * 2, 1)

        self.softmax = nn.Softmax(dim=-1)

        self.use_logic_attention = use_logic_attention
        self.use_nn_attention = use_nn_attention

        self.no_mask = no_mask
        if not self.no_mask:
            # 用pad虚拟邻居填充 aggregate的时候需要mask掉邻居
            # cnt_e 是取不到的 就是 0
            self.mask_emb = torch.cat([torch.ones([cnt_e, 1]), torch.zeros([1, 1])], 0).to(device)  # (cnt_e+1, 1)
            # 令不存在的邻居的权重是负无穷 后面用减法使得负无穷
            self.mask_weight = torch.cat([torch.zeros(cnt_e, 1), torch.ones([1, 1]) * 1e19], 0).to(device)  # (cnt_e+1, 1)

        self.init_values()
    def init_values(self):
        nn.init.xavier_normal_(self.w_r.weight)
        nn.init.xavier_normal_(self.attn_W.weight)
        nn.init.xavier_normal_(self.u_a.weight)
        nn.init.xavier_normal_(self.zq_emb.weight)

    def projection(self, e, w_r):
        norm2w_r = f.normalize(w_r, p=2, dim=-1)
        # norm2w_r = w_r / torch.norm(w_r, dim=-1, keepdim=True)  # 要求 w_r 2范数 为 1
        return e - torch.sum(e * norm2w_r, dim=-1, keepdim=True) * norm2w_r

    def get_attn(self, batch_nei_rid, batch_nei_e_Tr_emb, batch_nei_rw, batch_q_rid):
        """
        每个 e_id 的 neighbor 信息: nei_rid, nei_eid, nei_rw:
        :param batch_nei_rid:        (batch_size, max_neighbor)
        :param batch_nei_e_Tr_emb:      (batch_size, max_neighbor, dim)
        :param batch_nei_rw:         (batch_size, max_neighbor) 就是Logic的注意力权重了 在framework里求完了
        :param batch_q_rid:          (batch_size, )
        :return:
        """
        if self.use_logic_attention and not self.use_nn_attention:
            return batch_nei_rw

        max_neighbor = batch_nei_rid.shape[-1]

        # 1. 计算非标准化的NN注意力权重: alpha_{j|i,q}^{'}  :(batch_size, max_neighbor)
        #   a 获取 z_q  :(batch_size, max_neighbor, dim)
        batch_nei_q_rid = batch_q_rid.unsqueeze(-1).tile((1, max_neighbor))
        batch_z_q = self.zq_emb(batch_nei_q_rid)  # (batch_size, max_neighbor, dim)
        #   b 拼贴 z_q 和 tr_e
        concat_emb = torch.cat((batch_z_q, batch_nei_e_Tr_emb), dim=-1)  # (batch_size, max_neighbor, dim * 2)
        #   c 经过 W_a 层
        wa_concat = self.attn_W(concat_emb)  # (batch_size, max_neighbor, dim * 2)
        #   d 经过 tanh层
        tanh = torch.tanh(wa_concat)  # (batch_size, max_neighbor, dim * 2)
        #   e 和u_a 相乘
        _alpha = self.u_a(tanh).squeeze(-1)  # (batch_size, max_neighbor)
        if not self.no_mask:
            mask_logic = self.mask_weight[batch_nei_rid]  # (batch_size, max_neighbor, 1)
            _alpha -= mask_logic.squeeze(-1)

        # 2. 计算标准化的NN注意力权重
        alpha = self.softmax(_alpha)  # (batch_size, max_neighbor)

        # 3. 计算Logic+NN权重
        if self.use_logic_attention:
            attn = alpha + batch_nei_rw  # (batch_size, max_neighbor)
        else:
            attn = alpha
        return attn

    def forward(self, batch_nei_rid, batch_nei_e_emb, batch_nei_rw, batch_q_rid):
        # 1 获得 Tr(ej)   :(batch_size, max_neighbor, dim)
        #   1.1 获得虚拟的所有邻居ej的Tr_emb
        w_r = self.w_r(batch_nei_rid)  # :(batch_size, max_neighbor, dim)
        batch_nei_e_Tr_emb = self.projection(batch_nei_e_emb, w_r)  # :(batch_size, max_neighbor, dim)
        #   1.2 通过mask获得真实邻居的Tr_emb
        if not self.no_mask:
            mask = self.mask_emb[batch_nei_rid]  # (batch_size, max_neighbor, 1)
            batch_nei_e_Tr_emb = batch_nei_e_Tr_emb * mask

        # 2 获得注意力 alpha_Logic + alpha_NN    :(batch_size, max_neighbor)
        attn = self.get_attn(batch_nei_rid, batch_nei_e_Tr_emb, batch_nei_rw, batch_q_rid)

        # 3 获得 e_i^O = \sum attn * Tr(ej)    (batch_size, dim)
        e_out = torch.sum(attn.unsqueeze(-1) * batch_nei_e_Tr_emb, dim=-2)

        return e_out
