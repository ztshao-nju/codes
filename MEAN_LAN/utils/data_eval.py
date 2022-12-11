import random
import parser

from data_graph import load_triples
from options.args_hander import ARGs
from options.logger import logFrame
from eval import create_eval_data
import os
import copy
import torch
import time

def load_n_triples(path_list, logger):
    list_triplets = []
    for path in path_list:
        list_triplets.append(load_triples(path, logger))
    return tuple(list_triplets)

def load_data(data_dir, logger):
    train_triplets, aux_triplets, dev_triplets, test_triplets = load_n_triples(
        [os.path.join(data_dir, "train"),
         os.path.join(data_dir, "aux"),
         os.path.join(data_dir, "dev"),
         os.path.join(data_dir, "test"),],
        logger
    )
    return train_triplets, aux_triplets, dev_triplets, test_triplets

def save_eval_data(pos_triplets, true_triplets, cnt_e, cnt_r, mode_id, path, sample_pos_num=None, sample_true_num=None):
    """
    :param pos_triplets:    需要产生neg三元组的pos三元组们  list
    :param true_triplets:   正确的三元组们
    :param cnt_e:  实体数量
    :param mode_id: 表示neg三元组置换的位置 0是头实体 2是尾实体
    :return:
    """
    device = "cpu"
    eval_data = create_eval_data(pos_triplets, true_triplets, cnt_e, cnt_r, mode_id, device, logger,
                                 sample_pos_num=sample_pos_num, sample_true_num=sample_true_num)

    with open(path, "w") as f:
        for data in eval_data:
            for h, r, t in data:
                f.write('{} {} {} '.format(h, r, t))
            f.write('\n')
    print('over')
def read_eval_data(path, device):
    eval_data = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            triplets = line.strip().split(' ')
            triplets = [int(id) for id in triplets]
            eval_data.append(torch.tensor(triplets).view(-1, 3).to(device))
    return eval_data

if __name__ == '__main__':
    args_dir = os.path.join("..", "options", "json", "lan_LinkPredict.json")
    args = ARGs()
    args.load_args(args_dir)
    args.data_dir = os.path.join("..", "data", args.kg_dir, args.mode)
    # args.save_dir = os.path.join("checkpoints", args.save_dir)
    # args.log_dir = os.path.join("checkpoints", args.log_dir)

    log = logFrame()
    logger = log.getlogger(os.path.join("..", "checkpoints", "data_eval"))  # info控制台 debug文件

    from utils.data_graph import Graph
    g = Graph(args, logger)

    train_triplets, aux_triplets, dev_triplets, test_triplets = \
        load_data(os.path.join("..", "data", "fb15K", "head-10"), logger)
    true_for_valid = []
    true_for_test = train_triplets + aux_triplets + dev_triplets + test_triplets

    # test
    eval_data_path = os.path.join("eval_data")
    save_eval_data(test_triplets, true_for_test, g.cnt_e, g.cnt_r, 2, eval_data_path, sample_pos_num=1000, sample_true_num=1)
    # read_eval_data(eval_data_path, 'cpu')