import argparse
import time
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"  # 注意要放到 import torch 之前
import torch


# device = "cpu"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch.optim as optim

from options.args_hander import ARGs
from options.logger import logFrame

from utils.data_graph import Graph
from utils.data_pytorch import KGDataset, collate_kg, move_to_device
from torch.utils.data import DataLoader

from model.framework import Framework
from eval import evaluate
from utils.model_ckpt import save_checkpoint, load_checkpoint, create_file


# device 在get_params中确定
# 提供默认参数 args_dir提供json参数覆盖默认参数
def get_params(json_name=None):
    parser = argparse.ArgumentParser(description='Run Model MEAN or LAN')

    parser.add_argument('--json_name', '-J', type=str)
    parser.add_argument('--device', '-D', type=str, default='0')
    parser.add_argument('--kg_dir', '-K', type=str, default="fb15K")
    parser.add_argument('--mode', '-M', type=str, default="head-10")

    parser.add_argument('--experiment_name', '-E', type=str)
    parser.add_argument('--aggregate_type', '-A', type=str)  # mean/attention

    parser.add_argument('--use_logic_attention', type=int, default=0)
    parser.add_argument('--use_nn_attention', type=int, default=0)

    parser.add_argument('--max_neighbor', type=int, default=64)
    parser.add_argument('--dim', type=int, default=100)

    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--epoch_per_checkpoint', type=int, default=50)
    parser.add_argument('--num_neg', type=int, default=1)

    parser.add_argument('--predict_mode', type=str, default="head")
    parser.add_argument('--type', type=str)  # train/test

    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--checkpoint_path', type=str)

    args = ARGs(parser.parse_args())
    if json_name != None:
        args.json_name = json_name
    if args.json_name != None:
        args.json_name = os.path.join("options", "json", args.json_name + ".json")
        args.load_args(args.json_name)

    # args.output()
    args, device = set_path(args)

    return args, device

def set_path(args):
    # 组装其他信息
    args.data_dir = os.path.join("data", args.kg_dir, args.mode)

    args.save_dir = os.path.join("checkpoints", args.experiment_name)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    args.log_dir = os.path.join("logs", args.experiment_name)

    if args.checkpoint_path != None:
        args.checkpoint_path = os.path.join("checkpoints", args.experiment_name, args.checkpoint_path)
    assert not (args.resume == 1 and args.checkpoint_path == None)

    device = "cuda:" + args.device if torch.cuda.is_available() else "cpu"
    return args, device

def run_training(framework, optimizer, g, train_set, device, args, logger):
    all_epoch_loss = []
    b_num = len(train_set.dataset) // args.batch_size + (len(train_set.dataset) % args.batch_size != 0)
    logger.info('epoch:{} batch_size:{} batch_num:{} device:{} learning_rate:{} exp_name:{}'.format(
        args.num_epoch, args.batch_size, b_num, device, args.learning_rate, args.experiment_name))
    logger.info('================== start training ==================')
    start = time.time()
    best_performance = 0  # mrr

    start_epoch = 0
    EVALUATION = True
    if args.resume:  # 恢复断点
        start_epoch = load_checkpoint(args, framework, optimizer, logger)
    for curr_epoch in range(start_epoch, args.num_epoch):
        framework.train()

        curr_epoch_loss = 0
        for batch_pos_triplet, batch_neg_triplet in train_set:
            batch_pos_triplet = move_to_device(batch_pos_triplet, device)
            batch_neg_triplet = move_to_device(batch_neg_triplet, device)
            hp, rp, tp = batch_pos_triplet
            hn, rn, tn = batch_neg_triplet

            optimizer.zero_grad()
            loss = framework(hp, rp, tp, hn, rn, tn)  # 损失函数
            loss.backward()
            optimizer.step()

            curr_epoch_loss += loss.item()
        all_epoch_loss.append(curr_epoch_loss)
        epoch_t = time.time()
        content = '[curr epoch over] epoch:{} loss:{} time:{}'.format(curr_epoch, curr_epoch_loss, epoch_t - start)
        logger.debug(content)


        # 本轮结束 保存断点保存断点模型
        if (curr_epoch + 1) % args.epoch_per_checkpoint == 0:
            logger.info(content)
            save_checkpoint(framework, optimizer, curr_epoch, args.save_dir)

        if EVALUATION and (curr_epoch + 1) % args.epoch_per_checkpoint == 0:
            # 判断性能 并保存最好的模型
            hits_nums, mrr = evaluate(framework, g, 'train', logger, device)
            if mrr > best_performance:
                best_performance = mrr
                torch.save(framework.state_dict(), os.path.join(args.save_dir, "best"))
                logger.info('================== [best performance] ================== ')

    torch.save(framework.state_dict(), os.path.join(args.save_dir, "last"))

def process_data(args, logger):
    g = Graph(args, logger)
    dataset = KGDataset(g, args.num_neg, args.predict_mode, logger)
    train_set = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_kg)
    return g, dataset, train_set


def train():
    ##############################################################################################################
    # 1. 读入参数和log
    json_name = None
    args, device = get_params()
    log = logFrame()
    logger = log.getlogger(args.log_dir)  # info控制台 debug文件
    logger.info('================== 开始运行 ==================')
    logger.info('日志路径:{}'.format(args.log_dir))
    logger.info('JSON路径:{}'.format(json_name if json_name != None else 'None'))
    logger.info('TYPE:{}'.format(args.type))

    # print('log_dir:{}'.format(args.log_dir))
    ##############################################################################################################
    # 2. 处理数据集
    g, dataset, train_set = process_data(args, logger)

    ##############################################################################################################
    # 3. 模型 优化器
    framework = Framework(g, args, device)
    framework.to(device)
    # 通过 weight_decay 添加L2正则化
    optimizer = optim.Adam(framework.parameters(recurse=True), lr=args.learning_rate, weight_decay=args.weight_decay)

    ##############################################################################################################
    # 4. 训练 / 测试
    if args.type == "train":
        run_training(framework, optimizer, g, train_set, device, args, logger)

    if args.type == "test":
        model_path = os.path.join(args.save_dir, "best")
        logger.debug(' 加载模型 {} '.format(model_path))
        ckpt = torch.load(model_path)
        framework.load_state_dict(ckpt)
        framework.to(device)

    ##############################################################################################################
    # 5. 评估选择模型
    if args.type == "test":
        evaluate(framework, g, 'test', logger, device)


if __name__ == '__main__':
    train()