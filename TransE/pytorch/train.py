import torch
import torch.nn as nn
from data_process import load_triples, MyDataset
from torch.utils.data import Dataset, DataLoader
from transe import TransE

margin = 4
learning_rate = 0.01
n_epoch = 500
batch_size = 9600
dimension = 50
norm = 1

id2e, e2id, id2r, r2id, id_triples = load_triples()  # 读取信息
dataset = MyDataset(id_triples)
train_set = DataLoader(dataset, batch_size=batch_size, shuffle=True)
transe = TransE(id_triples, id2e, id2r, margin, dimension)
for batch in train_set:
    transe.create_neg_triples(batch)
