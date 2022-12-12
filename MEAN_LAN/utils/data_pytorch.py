import random

import torch
from torch.utils.data import Dataset, DataLoader
import copy
import sys
import numpy as np

from collections import defaultdict


class KGDataset(Dataset):

    def __init__(self, g, num_neg, predict_mode, logger):  # g是data_graph 中的 Graph
        # self.args = args
        self.num_neg = num_neg
        self.predict_mode = predict_mode
        self.logger = logger

        self.triplets = g.train_triplets
        self.cnt_e = g.cnt_e

        self.hr_t = None
        self.tr_h = None
        self.hr_t, self.tr_h = self.init_true(g.train_triplets + g.aux_triplets)

    def init_true(self, true_triplets):
        self.hr_t = defaultdict(list)
        self.tr_h = defaultdict(list)
        if self.predict_mode == 'head':
            for h, r, t in true_triplets:
                self.hr_t[(h, r)].append(t)
        return self.hr_t, self.tr_h

    def __getitem__(self, index):
        pos_triplet = self.triplets[index]
        neg_triplet = create_corrupt_triplets(self.cnt_e, pos_triplet, self.num_neg, self.predict_mode)
        return pos_triplet, neg_triplet

    def __len__(self):
        return len(self.triplets)


def split_triplet(triplets):  # Tensor (num, 3)
    triplets = torch.transpose(triplets, 0, 1)
    return triplets[0], triplets[1], triplets[2]


def collate_kg(data):  # {list:batch_size}
    batch_pos_triplets = torch.tensor([pos for pos, neg in data], dtype=torch.long)
    batch_neg_triplets = torch.tensor([neg for pos, neg in data], dtype=torch.long).view(-1, 3)
    return split_triplet(batch_pos_triplets), split_triplet(batch_neg_triplets)


def move_to_device(batch_triplet, device):
    h, r, t = batch_triplet
    h = h.to(device)
    r = r.to(device)
    t = t.to(device)
    return h, r, t


def create_corrupt_triplets(cnt_e, pos_triplet, corrupt_num, mode):
    neg_triplets = np.tile(np.asarray(copy.deepcopy(pos_triplet)), corrupt_num)
    for i in range(corrupt_num):
        base_id = i * 3
        if mode == 'head':  # 修改tail
            neg_triplets[base_id + 2] = random.randint(0, cnt_e - 1)
        elif mode == 'tail':
            neg_triplets[base_id + 0] = random.randint(0, cnt_e - 1)
    return neg_triplets.tolist()
