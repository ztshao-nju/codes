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

    def __init__(self, eval_triplets, answer_pool, cnt_e, predict_mode, logger):
        self.predict_mode = predict_mode
        self.logger = logger

        self.eval_triplets = eval_triplets
        self.answer_pool = answer_pool
        self.cnt_e = cnt_e  # 注意 with torch.no_grad(): 取消梯度计算可以降低内存
        self.hr_t = None
        self.tr_h = None

        self.hr_t= self.init_true()


    def init_true(self):
        self.hr_t = defaultdict(list)
        if self.predict_mode == 'head':
            for h, r, t in self.answer_pool:
                self.hr_t[(h, r)].append(t)
        return self.hr_t

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
            complet_labels[la+1] = True
        complet_labels[0] = False  # 注意是 False
        return complet_labels


def online_metric(hits_nums, model, eval_loader, device, logger):
    logger.info('================== start online evaluation ==================')
    start = time.time()
    nums = len(hits_nums)
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
        rank = (-batch_score).argsort(dim=-1).argmin(dim=-1) + 1.0  # position
        mrr += torch.sum(1.0 / rank)
        for id, hits in enumerate(hits_nums):
            ans[id] += torch.sum(rank <= hits).detach_()
        if (_index + 1) % 200 == 0:
            logger.debug('id:{} time:{}'.format(_index + 1, time.time() - start))
        if (_index + 1) % 1000 == 0:
            logger.info('id:{} time:{}'.format(_index + 1, time.time() - start))

    num = len(eval_loader.dataset)
    ans = [100.0 * v / num for v in ans]
    logger.info('================== end online evaluation:{} =================='.format(time.time() - start))

    return ans, mrr / num


def evaluate(framework, g, eval_type, logger, device):
    batch_size = 1  # 4
    num_workers = 0  # 不确定是不是它导致的debug卡住
    if eval_type == 'train':
        eval_triplets = g.dev_triplets
        answer_pool = g.train_triplets + g.aux_triplets
    elif eval_type == 'test':
        eval_triplets = g.test_triplets
        answer_pool = g.train_triplets + g.aux_triplets + g.dev_triplets + g.test_triplets

    eval_dataset = EvalDataset(eval_triplets, answer_pool, g.cnt_e, 'head', logger)
    dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    triplets_num = len(eval_dataset.eval_triplets)
    batch_num = (triplets_num // batch_size) + int(triplets_num % batch_size != 0)

    logger.info('{} evaluation: triplets_num:{}, batch_size:{}, batch_num:{}, cnt_e:{}, num_workers:{}'.format(
        eval_type, triplets_num, batch_size, batch_num, g.cnt_e, num_workers
    ))
    hits_nums, mrr = online_metric([1, 3, 10], framework, dataloader, device, logger)
    logger.info('hits@1:{:.6f} hits@3:{:.6f} hits@10:{:.6f} mrr:{:.6f}'.format(
        hits_nums[0].item(), hits_nums[1].item(), hits_nums[2].item(), mrr.item()))

    return hits_nums, mrr