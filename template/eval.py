from torch.utils.data import Dataset, DataLoader

import torch
import copy
import random
import numpy as np
import time
from collections import defaultdict


class EvalDataset(Dataset):

    def __init__(self, data):  # g是data_graph 中的 Graph
        # 赋值 处理 data
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]  # 自行处理
        return item

    def __len__(self):
        return len(self.data)

    # LAN 崔博提供的一种做法 通过张量处理加速运算，避免查找每个三元组是否已经是出现过的正三元组
    # def __getitem__(self, index):
    #     # TODO: 只包括了 'head' 模式的
    #     pos_triplet = self.eval_triplets[index]
    #     h, r, t = pos_triplet
    #     neg_triplets = [[h, r, eid] for eid in range(self.cnt_e)]
    #     triplets = [pos_triplet] + neg_triplets
    #     labels = self.get_label(self.hr_t[(h, r)])
    #     triplets = torch.tensor(triplets).transpose(0, 1)
    #     return triplets[0], triplets[1], triplets[2], labels
    #
    # def get_label(self, labels):
    #     complet_labels = torch.zeros(self.cnt_e + 1, dtype=torch.bool)
    #     for la in labels:
    #         complet_labels[la+1] = True
    #     complet_labels[0] = False  # 注意是 False
    #     return complet_labels


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

        with torch.no_grad():  # 注意不能漏掉这个 减少内存消耗
            model.eval()
            batch_score = model.task1_batch_score(h, r, t)

        # 计算排名的一种方式 0是预测三元组的下标 其他的是自己构造的负三元组
        # labels true表示是false neg三元组 值取负无穷 排序就排不到
        batch_score = torch.where(labels, -torch.ones_like(batch_score) * 100000.0, batch_score)
        batch_score = batch_score.view(eval_loader.batch_size, -1)
        rank = (-batch_score).argsort(dim=-1).argmin(dim=-1) + 1.0  # position
        mrr += torch.sum(1.0 / rank)
        for id, hits in enumerate(hits_nums):
            ans[id] += torch.sum(rank <= hits).detach_()

    num = len(eval_loader.dataset)
    ans = [100.0 * v / num for v in ans]
    logger.info('================== end online evaluation:{} =================='.format(time.time() - start))
    return ans, mrr / num


def evaluate(model, data, eval_type, logger, device):
    batch_size = 1  # 4
    num_workers = 0  # 不确定是不是它导致的debug卡住

    eval_data = None
    if eval_type == 'train':  # 验证阶段
        # 处理 eval_data
        pass
    elif eval_type == 'test':  # 测试阶段
        # 处理 eval_data
        pass

    eval_dataset = EvalDataset(eval_data)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    hits_nums, mrr = online_metric([1, 3, 10], model, eval_dataloader, device, logger)
    logger.debug('hits@1:{:.6f} hits@3:{:.6f} hits@10:{:.6f} mrr:{:.6f}'.format(
        hits_nums[0].item(), hits_nums[1].item(), hits_nums[2].item(), mrr.item()))
    return hits_nums, mrr