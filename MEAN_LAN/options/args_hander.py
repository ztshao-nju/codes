import argparse
import json
import sys
import time


class ARGs:
    def __init__(self, args=None):
        if args != None:
            for arg in vars(args):
                setattr(self, arg, getattr(args, arg))

    def load_args(self, args_dir):
        with open(args_dir, 'r') as f:
            args_dict = json.load(f)
            f.close()
        for k, v in args_dict.items():
            setattr(self, k, v)

    def output(self):
        for item in dir(self):
            if not item.startswith('__') and not item.startswith("output"):
                print(item, self.__getattribute__(item))

# path = '../pytorch/parse_args.json'
# args = load_args(path)
#
# time_info = '{}_log.txt'.format(time.strftime("%Y_%m_%d %H_%M_%S", time.localtime()))
# args.log_path = args.log_path + time_info
#
# # args.output()
#
# log = logFrame()
# logger = log.getlogger(args.log_path)
