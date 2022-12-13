import os.path
from options.args_hander import ARGs
from options.logger import logFrame

from utils.data_pytorch import KGDataset, collate_kg, move_to_device
from torch.utils.data import Dataset, DataLoader

import torch
import copy
import random
import numpy as np
import time
from collections import defaultdict
from utils.data_graph import Graph


class EvalDataset(Dataset):

    def __init__(self, eval_triplets, answer_pool, cnt_e, predict_mode, logger):  # g是data_graph 中的 Graph
        # self.args = args
        self.predict_mode = predict_mode
        self.logger = logger

        self.eval_triplets = eval_triplets
        self.answer_pool = answer_pool
        self.cnt_e = cnt_e  # 注意 with torch.no_grad(): 取消梯度计算可以降低内存
        self.hr_t = None
        self.tr_h = None
        self.hr_t, self.tr_h = self.init_true()

    def init_true(self):
        self.hr_t = defaultdict(list)
        self.tr_h = defaultdict(list)
        if self.predict_mode == 'head':
            for h, r, t in self.answer_pool:
                self.hr_t[(h, r)].append(t)
        return self.hr_t, self.tr_h

    def __getitem__(self, index):
        # TODO: 只包括了 'head' 模式的
        pos_triplet = self.eval_triplets[index]
        h, r, t = pos_triplet
        neg_triplets = [[h, r, eid] for eid in range(self.cnt_e)]
        triplets = [pos_triplet] + neg_triplets
        labels = self.get_label(self.hr_t[(h, r)])
        triplets = torch.tensor(triplets).transpose(0, 1)
        return triplets[0], triplets[1], triplets[2], labels

    def __len__(self):
        return len(self.eval_triplets)

    def get_label(self, labels):
        complet_labels = torch.zeros(self.cnt_e + 1, dtype=torch.bool)
        for la in labels:
            complet_labels[la] = True
        complet_labels[0] = False  # 注意是 False
        return complet_labels


def online_metric(hit_nums, model, eval_loader, device, logger):
    logger.info('================== start online evaluation ==================')
    start = time.time()
    nums = len(hit_nums)
    ans = [0.0 for i in range(nums)]
    mrr = 0
    for _index, (h, r, t, labels) in enumerate(eval_loader):
        h = h.to(device)
        r = r.to(device)
        t = t.to(device)
        labels = labels.to(device)  # labels 代表在当前批次中 是否是正样本

        with torch.no_grad():
            model.eval()
            batch_score = model.task1_batch_score(h, r, t)
        batch_score = torch.where(labels, -torch.ones_like(batch_score) * 100000.0, batch_score)
        batch_score = batch_score.view(eval_loader.batch_size, -1)
        pos = (-batch_score).argsort(dim=-1).argmin(dim=-1) + 1  # position
        mrr += 1.0 / pos
        for index, hit in enumerate(hit_nums):
            ans[index] += torch.sum(pos < hit).detach_()
        if (_index + 1) % 200 == 0:
            logger.debug('id:{} time:{}'.format(_index + 1, time.time() - start))
        if (_index + 1) % 800 == 0:
            logger.info('id:{} time:{}'.format(_index + 1, time.time() - start))

    num = len(eval_loader.dataset)
    ans = [100.0 * v / num for v in ans]
    logger.info('================== end online evaluation:{} =================='.format(time.time() - start))

    return ans, mrr / num
