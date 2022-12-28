import torch
from torch.utils.data import Dataset, DataLoader
import copy
import numpy as np
import random
from utils.data_helper import DataHelper

# 训练数据集
class KGDataset(Dataset):

    def __init__(self, data):  # g是data_graph 中的 Graph
        # 赋值 处理 data
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]  # 自行处理
        return item

    def __len__(self):
        return len(self.data)


# 在 DataLoader 中使用 __getitem__ 后进入 collate_kg
def collate_kg(data):  # {list:batch_size}
    # 自行处理
    return data

# train epoch训练中需要把 getitem 取出来的数据放到 device 上
def move_to_device(batch_triplet, device):
    h, r, t = batch_triplet
    h = h.to(device)
    r = r.to(device)
    t = t.to(device)
    return h, r, t


# 返回 dataset 和 dataloader
def process_data(args, logger):
    data_helper = DataHelper(args, logger)  # 通过 args 进入新的读取数据的函数 得到数据

    dataset = KGDataset(data_helper)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_kg)
    return data_helper, dataset, train_loader



