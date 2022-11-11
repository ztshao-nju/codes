import argparse
import json
import sys
import time
from python.szt_logger import logFrame


class ARGs:
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)

    def output(self):
        for item in dir(self):
            if not item.startswith('__') and not item.startswith("output"):
                print(item, self.__getattribute__(item))


def load_args(file_path):
    with open(file_path, 'r') as f:
        args_dict = json.load(f)
        f.close()
    # print("load arguments:", args_dict)
    args = ARGs(args_dict)
    return args


path = '../pytorch/parse_args.json'
args = load_args(path)

time_info = '{}_log.txt'.format(time.strftime("%Y_%m_%d %H_%M_%S", time.localtime()))
args.log_path = args.log_path + time_info

# args.output()

log = logFrame()
logger = log.getlogger(args.log_path)



# 文件信息
# parser = argparse.ArgumentParser(description='TransE Data Process')
# parser.add_argument('--prefix', type=str, default='..\\data\\freebase_mtr100_mte100-')
# parser.add_argument('--type', type=str, default='train')
# parser.add_argument('--log_path', type=str, default='..\\szt_loggings\\test\\{}_log.txt'.format(
#     time.strftime("%Y_%m_%d %H_%M_%S", time.localtime())))
# args = parser.parse_args()
#
#
# args.path = args.prefix + args.type + '.txt'
# args.eid_list_path = args.prefix + 'eid_list.txt'
# args.rid_list_path = args.prefix + 'rid_list.txt'
# print(args)

# parser.add_argument('--batch', type=int, default=4)
# parser.add_argument('--load_model', action='store_true',
#                     help='Load existing model?')
