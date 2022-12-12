import os.path
from options.args_hander import ARGs
from options.logger import logFrame

from utils.data_pytorch import KGDataset, collate_kg, move_to_device
from torch.utils.data import Dataset, DataLoader

import torch
import copy
import random
import time
from collections import defaultdict
from utils.data_graph import Graph


class EvalDataset(Dataset):

    def __init__(self, pos_triplets, cnt_e, predict_mode, logger):  # g是data_graph 中的 Graph
        # self.args = args
        self.predict_mode = predict_mode
        self.logger = logger

        self.pos_triplets = pos_triplets
        self.cnt_e = cnt_e  # 注意 with torch.no_grad(): 取消梯度计算可以降低内存
        self.hr_t = None
        self.tr_h = None
        self.hr_t, self.tr_h = self.init_true()
    def init_true(self):
        self.hr_t = defaultdict(list)
        self.tr_h = defaultdict(list)
        if self.predict_mode == 'head':
            for h, r, t in self.pos_triplets:
                if t < self.cnt_e:  # TODO: 为了处理内存太小控制的
                    self.hr_t[(h, r)].append(t)
        for hr in self.hr_t:
            self.hr_t[hr] = torch.tensor(self.hr_t[hr]) + 1
        return self.hr_t, self.tr_h

    def __getitem__(self, index):
        # TODO: 只包括了 'head' 模式的
        pos_triplet = self.pos_triplets[index]
        h, r, t = pos_triplet
        neg_triplets = [[h, r, eid] for eid in range(self.cnt_e)]
        labels = torch.zeros(self.cnt_e+1, dtype=torch.bool)
        labels[self.hr_t[(h, r)]] = True
        labels[0] = False  # 注意是 False
        # triplets = torch.tensor([pos_triplet]+neg_triplets).view(-1,3).transpose(0, 1)
        # labels = self.get_label(h, r)
        return [pos_triplet]+neg_triplets, labels  # 14296
        # return triplets[0], triplets[1], triplets[2], labels
    def __len__(self):
        return len(self.pos_triplets)

    def get_label(self, h, r):
        labels = torch.zeros(self.cnt_e + 1, dtype=torch.bool)
        labels[self.hr_t[(h, r)]] = True
        labels[0] = False  # 注意是 False
        return labels


def collate_eval(data):
    # for sample in data:
    #     for i in range(10):
    #         print(i)
    # triplets, labels = data[0]
    triplets = []
    labels = None
    for tr, la in data:
        triplets += tr
        if labels == None:
            labels = la
        else:
            labels = torch.cat((labels, la))
    triplets = torch.tensor(triplets).transpose(0, 1)
    return triplets[0], triplets[1], triplets[2], labels


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
            # model.eval()
            batch_score = model.task1_batch_score(h, r, t)
            # torch.cuda.empty_cache()
        batch_score = torch.where(labels, -torch.ones_like(batch_score) * 100000.0, batch_score)
        pos = (-batch_score).argsort().argmin() + 1
        mrr += 1.0 / pos
        for index, hit in enumerate(hit_nums):
            if pos <= hit:
                ans[index] += 1.0
        if (_index + 1) % 20 == 0:
            logger.info('id:{} time:{}'.format(_index+1, time.time()-start))

    num = len(eval_loader.dataset)
    ans = [100.0 * v / num for v in ans]
    logger.info('================== end online evaluation:{} =================='.format(time.time()-start))

    return ans, mrr / num


if __name__ == '__main__':
    args_dir = os.path.join("options", "json", "lan_LinkPredict.json")

    args = ARGs()
    args.load_args(args_dir)
    args.data_dir = os.path.join("data", args.kg_dir, args.mode)

    log = logFrame()
    logger = log.getlogger(os.path.join("checkpoints", "data_eval"))  # info控制台 debug文件

    g = Graph(args, logger)
    eval_dataset = EvalDataset(g.train_triplets+g.aux_triplets, g.cnt_e, 2, 'head', logger)
    dataloader = DataLoader(eval_dataset, collate_fn=collate_eval, shuffle=False, batch_size=2)
