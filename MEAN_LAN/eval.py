import torch
import copy

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
    ans = [v / len(eval_data) * 100.0 for v in ans]
    return ans, mrr / len(eval_data)

def create_eval_data(pos_triplets, true_triplets, cnt_e, mode_id, device, sample_pos_num=None, sample_true_num=None):
    """
    :param pos_triplets:    需要产生neg三元组的pos三元组们  list
    :param true_triplets:   正确的三元组们
    :param cnt_e:  实体数量
    :param mode_id: 表示neg三元组置换的位置 0是头实体 2是尾实体
    :return:
    """
    eval_data = []
    if sample_pos_num != None:
        pos_triplets = pos_triplets[:min(sample_pos_num, len(pos_triplets))]
    if sample_true_num != None:
        true_triplets = true_triplets[:min(sample_true_num, len(true_triplets))]
    # TODO: 内存不够了 测试 修改cnt_e小一些
    cnt_e = 2000
    # TODO: 记得删掉这块！

    for pos_triplet in pos_triplets:
        neg_triplets = []
        for eid in range(cnt_e):  # 1.5e4
            new_triplet = copy.deepcopy(pos_triplet)
            new_triplet[mode_id] = eid
            if new_triplet not in true_triplets:
                neg_triplets.append(new_triplet)
        eval_data.append(torch.tensor(pos_triplets+neg_triplets).view(-1, 3).to(device))
        # eval_data.append([pos_triplets, neg_triplets])
    return eval_data

def eval(type, framework, g, args, device, sample_pos_num=None, sample_true_num=None):
    mode_id = 2 if args.predict_mode == 'head' else 0  # head下修改tail
    if type == 'train':
        true_triplets_valid = g.train_triplets+g.aux_triplets
        data_list = create_eval_data(g.test_triplets, true_triplets_valid, g.cnt_e, mode_id, device,
                                     sample_pos_num=sample_pos_num, sample_true_num=sample_true_num)
    elif type == 'test':
        true_triplets_test = g.train_triplets + g.aux_triplets + g.dev_triplets + g.test_triplets
        data_list = create_eval_data(g.test_triplets, true_triplets_test, g.cnt_e, mode_id, device,
                                     sample_pos_num=sample_pos_num, sample_true_num=sample_true_num)
    hit_nums, mrr = metric([1, 3, 10], framework, data_list, device)
    return hit_nums, mrr
