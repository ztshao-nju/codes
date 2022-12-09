import os
import argparse
import torch

from options.args_hander import ARGs
from options.logger import logFrame

from utils.data_graph import Graph
from utils.data_pytorch import KGDataset, collate_kg, move_to_device
from torch.utils.data import DataLoader

from model.framework import Framework

# 提供默认参数 args_dir提供json参数可以覆盖默认参数
def get_params(args_dir=None):
    parser = argparse.ArgumentParser(description='Run Model MEAN or LAN')

    parser.add_argument('--data_dir', '-D', type=str, default="data\\fb15K\\head-10")
    parser.add_argument('--save_dir', '-S', type=str, default="checkpoints")
    parser.add_argument('--log_dir', '-L', type=str, default="checkpoints\\log_info")

    parser.add_argument('--aggregate_type', type=str, default="gnn_mean")
    parser.add_argument('--use_logic_attention', type=int, default=0)
    parser.add_argument('--use_nn_attention', type=int, default=0)

    parser.add_argument('--max_neighbor', type=int, default=64)
    parser.add_argument('--dim', type=int, default=100)

    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epoch_per_checkpoint', type=int, default=50)
    parser.add_argument('--num_neg', type=int, default=1)

    parser.add_argument('--loss_function', type=str, default="margin")
    parser.add_argument('--score_function', type=str, default="TransE")
    parser.add_argument('--corrupt_mode', type=str, default="partial")
    parser.add_argument('--predict_mode', type=str, default="head")
    parser.add_argument('--type', type=str, default="train")

    args = ARGs(parser.parse_args())
    if args_dir != None:
        args.load_args(args_dir)
    return args


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 1. 读入参数和log
    args_dir = os.path.join("options", "mean_LinkPredict.json")
    args = get_params(args_dir)
    log = logFrame()
    logger = log.getlogger(args.log_dir)  # info控制台 debug文件

    # 2. 处理数据集
    g = Graph(args, logger)
    dataset = KGDataset(args, g, logger)
    train_set = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_kg)

    # 3. 模型
    framework = Framework(g, args, device)
    framework.to(device)

    # 4. 训练
    for batch_pos_triplet, batch_neg_triplet in train_set:
        batch_pos_triplet = move_to_device(batch_pos_triplet, device)
        batch_neg_triplet = move_to_device(batch_neg_triplet, device)
        framework(batch_pos_triplet, batch_neg_triplet)
        print('ahahah1')


    # 4. 评估选择模型



