import argparse
import time
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"  # 注意要放到 import torch 之前
import torch
import torch.optim as optim


from train import get_params, set_path, process_data, run_training, evaluate
from options.logger import logFrame
from model.framework import Framework


def output_best_param(best_param, logger):
    logger.info('============ best param ============')
    logger.info('dim:{}  margin:{}'.format(best_param['dim'], best_param['margin']))
    hits_nums = best_param['hits']
    mrr = best_param['mrr']
    logger.info('hits@1:{:.6f} hits@3:{:.6f} hits@10:{:.6f} mrr:{:.6f}'.format(
        hits_nums[0].item(), hits_nums[1].item(), hits_nums[2].item(), mrr.item()))


if __name__ == '__main__':
    args, device = get_params()
    log_all = logFrame()
    logger = log_all.getlogger(args.log_dir)

    logger.info('====== grid search ======')
    logger.info('log_dir:{}'.format(args.log_dir))
    logger.info('aggregate_type:{}'.format(args.aggregate_type))

    g, dataset, train_set = process_data(args, logger)

    best_mrr = 0
    best_param = {
        "dim": 0,
        "margin": 0,
        "weight_decay": 0,
        "mrr": None,
        "hits": None
    }
    FINISH = []
    # for dim in [50, 100, 200]:
    # for margin in [1.0, 1.5, 2.0, 4.0]:
    for margin in [1.0, 1.5, 2.0]:
        dim = args.dim
        args.margin = margin
        args.experiment_name = 'd{}_m{}'.format(dim, margin)
        if args.experiment_name in FINISH:
            continue
        logger.info('======================== {} begin ========================'.format(args.experiment_name))

        args, device = set_path(args)

        framework = Framework(g, args, device)
        framework.to(device)
        optimizer = optim.Adam(framework.parameters(recurse=True), lr=args.learning_rate, weight_decay=args.weight_decay)

        run_training(framework, optimizer, g, train_set, device, args, logger)
        hits_nums, mrr = evaluate(framework, g, 'test', logger, device)

        logger.info('======================== {} over ========================'.format(args.experiment_name))
        logger.info('hits@1:{:.6f} hits@3:{:.6f} hits@10:{:.6f} mrr:{:.6f}'.format(
            hits_nums[0].item(), hits_nums[1].item(), hits_nums[2].item(), mrr.item()))

        if mrr.item() > best_mrr:
            best_mrr = mrr.item()
            best_param = {
                "dim": dim,
                "margin": margin,
                "mrr": mrr,
                "hits": hits_nums
            }
            output_best_param(best_param, logger)



    logger.info('============ grid search over ============')
    output_best_param(best_param, logger)
