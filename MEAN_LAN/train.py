import copy
import argparse
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 注意要放到 import torch 之前
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
import torch

# torch.cuda.set_device(2)
device = "cuda:1" if torch.cuda.is_available() else "cpu"

import torch.optim as optim
import numpy as np

from options.args_hander import ARGs
from options.logger import logFrame

from utils.data_graph import Graph
from utils.data_pytorch import KGDataset, collate_kg, move_to_device
from torch.utils.data import DataLoader

from model.framework import Framework
from eval import online_metric, EvalDataset
from utils.model_ckpt import save_checkpoint, load_checkpoint


# device = "cpu"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 提供默认参数 args_dir提供json参数覆盖默认参数
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

    parser.add_argument('--predict_mode', type=str, default="head")
    parser.add_argument('--type', type=str, default="train")
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--checkpoint_path', type=str)

    args = ARGs(parser.parse_args())
    if args_dir != None:
        args.load_args(args_dir)
    # args.output()
    args.data_dir = os.path.join("data", args.kg_dir, args.mode)
    args.save_dir = os.path.join("checkpoints", args.save_dir)
    # args.log_dir = os.path.join("logs", args.log_dir)
    if args.type == 'train':
        args.log_dir = os.path.join("logs", args.log_dir + '_train')
    elif args.type == 'test':
        args.log_dir = os.path.join("logs", args.log_dir + '_test')
    if args.checkpoint_path != None:
        args.checkpoint_path = os.path.join("checkpoints", args.checkpoint_path)

    assert not (args.resume == 1 and args.checkpoint_path == None)
    return args


def run_training(framework, args, logger):
    all_epoch_loss = []
    b_num = len(dataset) // args.batch_size + (len(dataset) % args.batch_size != 0)
    logger.info('epoch:{} batch_size:{} batch_num:{} device:{}'.format(args.num_epoch, args.batch_size, b_num, device))
    logger.info('================== start training ==================')
    start = time.time()
    best_performance = 0  # mrr

    start_epoch = 0
    EVALUATION = True
    if args.resume:  # 恢复断点
        start_epoch = load_checkpoint(args, framework, optimizer, logger)
    for curr_epoch in range(start_epoch + 1, args.num_epoch):
        framework.train()

        curr_epoch_loss = 0
        batch_id = 0
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
            batch_t = time.time()
            batch_id += 1
            if batch_id % 20 == 0:
                logger.debug('epoch:{} batch:{} loss:{} time:{}'.format(curr_epoch, batch_id, loss,
                                                                        batch_t - start))
        all_epoch_loss.append(curr_epoch_loss)
        epoch_t = time.time()
        logger.info('[curr epoch over] epoch:{} loss:{} time:{}'.format(curr_epoch, curr_epoch_loss, epoch_t - start))

        # 本轮结束 保存断点保存断点模型
        if (curr_epoch + 1) % args.epoch_per_checkpoint == 0:
            save_checkpoint(framework, optimizer, curr_epoch)

        if EVALUATION and (curr_epoch + 1) % args.epoch_per_checkpoint == 0:
            # 判断性能
            answer_pool = g.train_triplets + g.aux_triplets
            eval_dataset = EvalDataset(g.dev_triplets, answer_pool, g.cnt_e, 'head', logger)

            b_size = 1
            num = len(eval_dataset)  # 三元组数量
            b_num = (num // b_size) + int(num % b_size != 0)

            dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=b_size)
            hit_nums, mrr = online_metric([1, 3], framework, dataloader, device, logger)
            logger.info(
                'valid: triplets_num:{}, batch_size:{}, batch_num:{}, cnt_e:{}'.format(num, b_size, b_num, g.cnt_e))
            logger.info('[curr performance] epoch:{} loss:{} mrr:{:.6f} time:{}'.format(
                curr_epoch, curr_epoch_loss, mrr.item(), time.time() - start
            ))

            # 最好的模型 保存
            if mrr > best_performance:
                best_performance = mrr
                torch.save(framework.state_dict(), args.save_dir + str(curr_epoch))
                logger.info('================== [best performance] ================== ')

    torch.save(framework.state_dict(), args.save_dir)


if __name__ == '__main__':
    ##############################################################################################################
    # 1. 读入参数和log
    args_dir = os.path.join("options", "json", "lan_LinkPredict.json")
    args = get_params(args_dir)
    log = logFrame()
    logger = log.getlogger(os.path.join(args.log_dir))  # info控制台 debug文件

    ##############################################################################################################
    # 2. 处理数据集
    g = Graph(args, logger)
    dataset = KGDataset(g, args.num_neg, args.predict_mode, logger)
    train_set = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_kg)

    ##############################################################################################################
    # 3. 模型 优化器
    framework = Framework(g, args, device)
    framework.to(device)
    # 通过 weight_decay 添加L2正则化
    optimizer = optim.Adam(framework.parameters(recurse=True), lr=args.learning_rate, weight_decay=args.weight_decay)

    ##############################################################################################################
    # 4. 训练 / 测试
    if args.type == "train":
        run_training(framework, args, logger)

    if args.type == "test":
        model_path = args.save_dir + "999"
        logger.info(' 加载模型 {} '.format(model_path))
        ckpt = torch.load(model_path)
        framework.load_state_dict(ckpt)
        framework.to(device)

    ##############################################################################################################
    # 5. 评估选择模型
    if args.type == "test":
        batch_size = 1
        answer_pool = g.train_triplets + g.aux_triplets + g.dev_triplets + g.test_triplets
        eval_dataset = EvalDataset(g.dev_triplets, answer_pool, g.cnt_e, 'head', logger)
        dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size)
        triplets_num = len(eval_dataset.eval_triplets)
        batch_num = (triplets_num // batch_size) + int(triplets_num % batch_size != 0)
        logger.info('test evaluation: triplets_num:{}, batch_size:{}, batch_num:{}, cnt_e:{}'.format(
            triplets_num, batch_size, batch_num, g.cnt_e
        ))
        framework.eval()
        hit_nums, mrr = online_metric([1, 3, 10], framework, dataloader, device, logger)
        logger.info('hit@1:{:.6f} hit@3:{:.6f} hit@10:{:.6f} mrr:{:.6f}'.format(
            hit_nums[0].item(), hit_nums[1].item(), hit_nums[2].item(), mrr.item()))
