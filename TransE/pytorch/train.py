import time

import torch
import torch.nn as nn
from data_process import load_triples, MyDataset
from torch.utils.data import Dataset, DataLoader
from transe import TransE
from python.szt_arg import args, logger
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"
id2e, e2id, id2r, r2id, id_triples = load_triples()  # read information
dataset = MyDataset(id_triples)
train_set = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
transe = TransE(id_triples, id2e, id2r, args.margin, args.dimension, device).to(device)
optim = torch.optim.SGD(transe.parameters(), lr=args.lr)
on_gpu = next(transe.parameters()).is_cuda
logger.info('Model Is On: %s' % "GPU" if on_gpu else "CPU")


def train(model, epoch, optimizer, tr_set):
    """
    :param model:
    :param epoch:
    :param optimizer:
    :param tr_set:
    :return:
    """

    start = time.time()
    loss_array = []
    # initialize relations, entities; norm2 relations
    model.my_initialize()
    for curr_epoch in range(epoch):
        model.train()
        # norm2 entities
        model.norm2_entities()

        epoch_loss = 0
        triple_num = 0

        for pos_triples in tr_set:
            curr_triple_num = len(pos_triples[0])
            optimizer.zero_grad()  # clear grad
            neg_triples = model.create_neg_triples(pos_triples)

            curr_loss = model(pos_triples, neg_triples)
            curr_loss.backward()
            optimizer.step()

            epoch_loss += curr_loss * curr_triple_num
            triple_num = triple_num + curr_triple_num
            # logger.info('curr_loss:%f, curr_triple_num:%d' % (curr_loss, curr_triple_num))

        # logger.info('epoch_loss:%f, total_data_num:%d' % (epoch_loss, triple_num))
        epoch_loss /= triple_num
        logger.info('epoch:%d loss:%f time:%f' % (curr_epoch, epoch_loss, time.time() - start))
        loss_array.append(epoch_loss)

    fig = plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(loss_array) + 1), loss_array, label='Train Loss')
    plt.show()
    fig.savefig('TransE_loss_plot.png', bbox_inches='tight')


train(transe, args.n_epoch, optim, train_set)
