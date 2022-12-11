import os.path
from options.args_hander import ARGs
from options.logger import logFrame

import torch
import copy
import random
import time

def metric(hit_nums, model, eval_data, device):
    nums = len(hit_nums)
    ans = [0.0 for i in range(nums)]
    mrr = 0
    for data in eval_data:
        # data: (num, 3)
        h = data.transpose(0,1)[0]
        r = data.transpose(0,1)[1]
        t = data.transpose(0,1)[2]
        batch_score = model.task1_batch_score(h, r, t)
        pos = (-batch_score).argsort().argmin() + 1
        mrr += 1.0 / pos
        for index, hit in enumerate(hit_nums):
            if pos <= hit:
                ans[index] += 1.0
    ans = [100.0 * v / len(eval_data) for v in ans]
    return ans, mrr / len(eval_data)

def eval(type, framework, g, args, device, logger, sample_pos_num=None, sample_true_num=None):
    if type == 'train':
        logger.info('train: create eval data from dev_triplets')
        true_triplets_valid = g.train_triplets+g.aux_triplets
        eval_data = create_eval_data(g.dev_triplets, true_triplets_valid, g.cnt_e, g.cnt_r, args.predict_mode,
                                     device, logger,
                                     sample_pos_num=sample_pos_num, sample_true_num=sample_true_num)
    elif type == 'test':
        test_from_file = True
        if test_from_file:
            logger.info('test: load eval data created from dev_triplets')
            eval_data = read_eval_data(os.path.join("utils", "eval_data"), device)
        else:
            logger.info('test: create eval data from dev_triplets')
            true_triplets_test = g.train_triplets + g.aux_triplets + g.dev_triplets + g.test_triplets
            eval_data = create_eval_data(g.test_triplets, true_triplets_test, g.cnt_e, g.cnt_r, args.predict_mode,
                                         device, logger,
                                         sample_pos_num=sample_pos_num, sample_true_num=sample_true_num)

    hit_nums, mrr = metric([1, 3, 10], framework, eval_data, device)
    return hit_nums, mrr

def create_eval_data(pos_triplets, true_triplets, cnt_e, cnt_r, mode, device, logger, sample_pos_num=None, sample_true_num=None):
    eval_data = create_list_data(pos_triplets, true_triplets, cnt_e, cnt_r, mode, logger,
                     sample_pos_num=sample_pos_num, sample_true_num=sample_true_num)
    eval_data = eval_data2tensor(eval_data, device)
    return eval_data
def read_eval_data(path, device):
    eval_data = []
    with open(path, "r") as f:
        lines = f.readlines()
        id = 0
        for line in lines:
            triplets = line.strip().split(' ')
            triplets = [int(id) for id in triplets]
            # print(id)
            eval_data.append(torch.tensor(triplets).view(-1, 3).to(device))
            id += 1
    return eval_data
def create_list_data(pos_triplets, true_triplets, cnt_e, cnt_r, mode, logger, sample_pos_num=None, sample_true_num=None):
    eval_data = []
    if sample_pos_num != None:
        pos_triplets = pos_triplets[:min(sample_pos_num, len(pos_triplets))]
    if sample_true_num != None:
        true_triplets = true_triplets[:min(sample_true_num, len(true_triplets))]

    # TODO: 内存不够了 测试 修改cnt_e小一些
    # cnt_e = 2000
    # sample_num 表示每个pos三元组生成的neg三元组的数量
    # sample_num = 100  # cnt_e
    # sample_id_list = random.sample(range(cnt_e), sample_num)
    sample_id_list = range(cnt_e)
    # TODO: 记得删掉这块！

    num = 0
    start = time.time()
    logger.info('================== create evaluation data ==================')
    for _index, pos_triplet in enumerate(pos_triplets):
        h, r, t = copy.deepcopy(pos_triplet)
        if r >= cnt_r:
            continue
        neg_id_set = set(sample_id_list)
        curr_line = [pos_triplet]
        for eid in sample_id_list:  # 1.5e4
            if mode == 'head':
                if [h, r, eid] not in true_triplets:
                    neg_id_set.discard(eid)
            elif mode == 'tail':
                if [eid, r, t] not in true_triplets:
                    neg_id_set.discard(eid)
        if mode == 'head':
            curr_line.extend([h, r, eid] for eid in neg_id_set)
        elif mode == 'tail':
            curr_line.extend([eid, r, t] for eid in neg_id_set)
        eval_data.append(curr_line)
        if (_index + 1) % 1 == 0:
            logger.info('负样本数据生成进度:{}/{}    时间:{}'.format(num, len(pos_triplets), time.time() - start))

    return eval_data
def eval_data2tensor(eval_data, device):
    eval_data_tensor = []
    for data in eval_data:
        curr_line = torch.tensor(data).view(-1, 3).to(device)
        eval_data_tensor.append(curr_line)
    return eval_data_tensor
def save_data(eval_data, path):
    with open(path, "w") as f:
        for data in eval_data:
            for h, r, t in data:
                f.write('{} {} {} '.format(h, r, t))
            f.write('\n')
    print('over')

if __name__ == '__main__':
    args_dir = os.path.join("options", "json", "lan_LinkPredict.json")

    args = ARGs()
    args.load_args(args_dir)
    args.data_dir = os.path.join("data", args.kg_dir, args.mode)

    log = logFrame()
    logger = log.getlogger(os.path.join("checkpoints", "data_eval"))  # info控制台 debug文件

    from utils.data_graph import Graph
    g = Graph(args, logger)

    train_triplets, aux_triplets, dev_triplets, test_triplets = \
        g.train_triplets, g.aux_triplets, g.dev_triplets, g.test_triplets
    true_for_valid = train_triplets + aux_triplets
    true_for_test = train_triplets + aux_triplets + dev_triplets + test_triplets

    # test
    eval_data_path = os.path.join("eval_data")
    eval_data = create_list_data(test_triplets, true_for_test, g.cnt_e, g.cnt_r,'head', logger, sample_pos_num=5, sample_true_num=None)
    save_data(eval_data, eval_data_path)
