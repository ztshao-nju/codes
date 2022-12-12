import copy
import os
import argparse
import time
import torch
import torch.optim as optim

from options.args_hander import ARGs
from options.logger import logFrame

from utils.data_graph import Graph
from utils.data_pytorch import KGDataset, collate_kg, move_to_device
from torch.utils.data import DataLoader

from model.framework import Framework
from eval import online_metric, EvalDataset, collate_eval

os.environ['CUDA_VISIBLE_DEVICES']='1, 2'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
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

    args = ARGs(parser.parse_args())
    if args_dir != None:
        args.load_args(args_dir)
    # args.output()
    args.data_dir = os.path.join("data", args.kg_dir, args.mode)
    args.save_dir = os.path.join("checkpoints", args.save_dir)
    args.log_dir = os.path.join("checkpoints", args.log_dir)
    return args


def run_training(framework, args, logger):
    start = time.time()
    all_epoch_loss = []
    batch_num = len(dataset) // args.batch_size + (len(dataset) % args.batch_size != 0)
    logger.info('epoch:{}    batch_size:{}    batch_num:{}    device:{}'.format(
        args.num_epoch, args.batch_size, batch_num, device)
    )
    logger.info('================== start training ==================')
    best_performance = 0  # mrr

    for curr_epoch in range(args.num_epoch):
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
                content = 'epoch:{} batch:{} loss:{:.6f} time:{:.1f}'.format(curr_epoch, batch_id, loss,
                                                                              batch_t - start)
                logger.debug(content)

        all_epoch_loss.append(curr_epoch_loss)
        epoch_t = time.time()
        logger.info(
            '[curr epoch over] epoch:{} loss:{:.6f} time:{:.1f}'.format(curr_epoch, curr_epoch_loss, epoch_t - start))

        # if (curr_epoch + 1) % args.epoch_per_checkpoint == 0:
        #     framework.eval()
        #     # 判断性能 保留最好的性能
        #     eval_dataset = EvalDataset(g.train_triplets + g.aux_triplets, g.cnt_e, 'head', logger)
        #     dataloader = DataLoader(eval_dataset, collate_fn=collate_eval, shuffle=False, batch_size=2)
        #     torch.cuda.empty_cache()
        #     with torch.no_grad():
        #         hit_nums, mrr = online_metric([1, 3], framework, dataloader, device, logger)
        #     logger.info('[curr performance] epoch:{} loss:{:.6f} mrr:{:3f} time:{:.1f}'.format(
        #         curr_epoch, curr_epoch_loss, mrr, time.time() - start
        #     ))
        #     if mrr > best_performance:
        #         best_performance = mrr
        #         torch.save(framework.state_dict(), args.save_dir)
        #         torch.save(framework.state_dict(), args.save_dir + str(curr_epoch))
        #         logger.info('[best performance] epoch:{} loss:{:.6f} mrr:{:3f} time:{:.1f}'.format(
        #             curr_epoch, curr_epoch_loss, mrr, time.time() - start
        #         ))
    torch.save(framework.state_dict(), args.save_dir)

if __name__ == '__main__':
    ##############################################################################################################
    # 1. 读入参数和log
    args_dir = os.path.join("options", "json", "lan_LinkPredict.json")
    args = get_params(args_dir)
    log = logFrame()
    logger = log.getlogger(os.path.join("checkpoints", "log_info"))  # info控制台 debug文件

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
        # torch.save(framework.state_dict(), args.save_dir)

    if args.type == "test":
        model_path = args.save_dir + ""
        logger.info(' 加载模型 {} '.format(model_path))
        ckpt = torch.load(model_path)
        framework.load_state_dict(ckpt)
        framework.to(device)

    ##############################################################################################################

    # import torchstat
    # # from torchstat import stat
    # torchstat.stat(framework, [(1024,), (1024,), (1024,), (1024,), (1024,), (1024,)])

    # 5. 评估选择模型
    if args.type == "test":
        # import torchsummary
        # torchsummary.summary(framework.cuda(), (3, 224, 224))


        # import torchsummary
        #
        # summary = torchsummary.summary(framework.cuda(),
        #                                input_size=[(1024,), (1024,), (1024,), (1024,), (1024,), (1024,)],
        #                                dtypes=[torch.long, torch.long, torch.long, torch.long, torch.long, torch.long])
        # print(summary)



        # logger.info('================== start evaluation ==================')
        batch_size = 2
        eval_dataset = EvalDataset(g.train_triplets + g.aux_triplets, g.cnt_e, 'head', logger)
        # dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size)
        dataloader = DataLoader(eval_dataset, collate_fn=collate_eval, shuffle=False, batch_size=batch_size)
        triplets_num = len(eval_dataset.pos_triplets)
        batch_num = triplets_num // batch_size + int(triplets_num % triplets_num != 0)
        logger.info('test evaluation: triplets_num:{}, batch_size:{}, batch_num:{}'.format(
            triplets_num, batch_size, batch_num
        ))
        framework.eval()
        hit_nums, mrr = online_metric([1, 3], framework, dataloader, device, logger)
        logger.info('hit@1:{:.6f} hit@3:{:.6f} hit@10:{:.6f} mrr:{:.6f}'.format(*tuple(hit_nums), mrr))
