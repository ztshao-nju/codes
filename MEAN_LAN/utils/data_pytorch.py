import random

import torch
from torch.utils.data import Dataset, DataLoader
import copy
import sys
import numpy as np

class KGDataset(Dataset):

    def __init__(self, args, g, logger):  # g是data_graph 中的 Graph
        self.args = args
        self.predict_mode = args.predict_mode
        self.g = g
        self.logger = logger

    def __getitem__(self, index):
        pos_triplet = self.g.train_triplets[index]
        neg_triplet = np.tile(np.asarray(copy.deepcopy(pos_triplet)), self.args.num_neg)
        for i in range(self.args.num_neg):
            base_id = i * 3
            if self.predict_mode == 'head':  # 修改tail
                neg_triplet[base_id + 2] = random.randint(0, self.g.cnt_e - 1)
            elif self.predict_mode == 'tail':
                neg_triplet[base_id + 0] = random.randint(0, self.g.cnt_e - 1)
            else:
                self.logger.info('没有识别出 KGDataset 的 predict_mode')
                sys.exit(0)
        neg_triplet = neg_triplet.tolist()
        return pos_triplet, neg_triplet

    def __len__(self):
        return len(self.g.train_triplets)

def split_triplet(triplets):  # Tensor (num, 3)
    triplets = torch.transpose(triplets, 0, 1)
    return triplets[0], triplets[1], triplets[2]

def collate_kg(data):   # {list:batch_size}
    batch_pos_triplets = torch.tensor([pos for pos, neg in data], dtype=torch.long)
    batch_neg_triplets = torch.tensor([neg for pos, neg in data], dtype=torch.long).view(-1, 3)
    return split_triplet(batch_pos_triplets), split_triplet(batch_neg_triplets)

def move_to_device(batch_triplet, device):
    h, r, t = batch_triplet
    h = h.to(device)
    r = r.to(device)
    t = t.to(device)
    return h, r, t