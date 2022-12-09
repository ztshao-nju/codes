import os
import argparse
from options.args_hander import ARGs
from options.logger import logFrame




# 提供默认参数 args_dir提供json参数
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
    parser.add_argument('--type', type=str, default="train")

    args = ARGs(parser.parse_args())
    if args_dir != None:
        args.load_args(args_dir)
    return args



if __name__ == '__main__':
    # 1. 读入参数和log
    args_dir = os.path.join("options", "mean_LinkPredict.json")
    args = get_params(args_dir)
    log = logFrame()
    logger = log.getlogger(args.log_dir)  # info控制台 debug文件



