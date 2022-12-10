from data_graph import load_triples
from options.args_hander import ARGs
from options.logger import logFrame
import os

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

# def create_true_pool(pos_triplets_list, mode_id):


if __name__ == '__main__':
    data_dir = ""
    log = logFrame()
    logger = log.getlogger(os.path.join("..", "checkpoints", "data_eval"))  # info控制台 debug文件
    train_triplets, aux_triplets, dev_triplets, test_triplets = \
        load_data(os.path.join("..", "data", "fb15K", "head-10"), logger)
    true_for_valid = []
    true_for_test = []

