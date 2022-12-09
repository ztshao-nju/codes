import numpy as np
import os
from collections import defaultdict

# 加载 paid_pathth 路径中 item和id之间的map
def load_id_map(id_path):
    id2item, item2id = {}, {}
    with open(id_path, 'r') as f:
        contents = f.readlines()
        for content in contents:
            item, index = content.strip().split('\t')
            index = int(index)
            id2item[index] = item
            item2id[item] = int(index)
    return id2item, item2id

# 读取 triplet_path 路径中的三元组
def load_triples(triplet_path, logger):
    _id_triples = []
    with open(triplet_path, 'r') as f:
        triples = f.readlines()
        for triple in triples:
            line = triple.strip().split('\t')
            line = [int(item) for item in line]
            h, r, t = line
            _id_triples.append([h, r, t])
    logger.info('[加载三元组] {} 数据集 三元组数目: {}'.format(triplet_path, len(_id_triples)))
    return _id_triples

# 读取所有数据
def load_data(args, logger):
    logger.info('================== 加载数据 ==================')
    # 1. 加载 实体和id 关系和id 之间的map
    id2e, e2id = load_id_map(os.path.join(args.data_dir, "entity2id.txt"))
    id2r, r2id = load_id_map(os.path.join(args.data_dir, "relation2id.txt"))
    logger.info('[加载实体和关系] 实体个数:{} 关系个数:{}'.format(len(id2e), len(id2r)))

    # 2. 加载4个三元组列表: train aux dev test
    train_triplets = load_triples(os.path.join(args.data_dir, "train"), logger)
    aux_triplets = load_triples(os.path.join(args.data_dir, "aux"), logger)
    dev_triplets = load_triples(os.path.join(args.data_dir, "dev"), logger)
    test_triplets = load_triples(os.path.join(args.data_dir, "test"), logger)

    return id2e, e2id, id2r, r2id, \
           train_triplets, aux_triplets, dev_triplets, test_triplets

class Graph:
    # 处理数据
    def __init__(self, args, logger):
        self.max_neighbor = args.max_neighbor

        self.cnt_tr_e = 0  # train 实体数量
        self.cnt_e = 0  # train+aux 实体数量
        self.cnt_r = 0  # 关系数量(没算上逆关系)
        self.corr = None  # np (cnt_r*2+1, cnt_r*2+1) self.corr[j][i] 表示P(ri->rj)
        self.graph = None  # dict{list([rid,tid,weight],,)}
        self.train_g = None  # np (cnt_e, max_neighbor, 2)  0-r_id 1-e_id
        self.train_w = None  # np (cnt_e, max_neighbor) weight-denominator
        # 1. 读取数据
        self.id2e, self.e2id, self.id2r, self.r2id, \
            train_triplets, aux_triplets, dev_triplets, test_triplets = \
            load_data(args, logger)

        # 2.3. 获取cnt数据 + 构造graph + 计算r之间的correlation(train,没有aux)
        self.get_cnt_graph(train_triplets, aux_triplets, args.max_neighbor, args, logger)

        # 4. 计算 Logic 权重分母 self.graph[i][j][2] 表示实体ei第j条邻居信息的r的Logic分母
        self.graph = self.calcu_max_denominator()

        # 5. 构造 train_g train_w
        self.train_g, self.train_w = self.calcu_neighbor()

        # 6. 重新整理 dev_triplets, test_triplets
        self.dev_triplets = self.reconstruct_triples(dev_triplets)
        self.test_triplets = self.reconstruct_triples(test_triplets)
        self.train_triplets = train_triplets

    # 6. 重新整理 dev_triplets, test_triplets
    def reconstruct_triples(self, triplets):
        new_triplets = []
        for h, r, t in triplets:
            if h >= self.cnt_e or t >= self.cnt_e:  # 注意是 cnt_e 不是 cnt_tr_e
                continue
            if r >= self.cnt_r:
                continue
            new_triplets.append([h, r, t])
        return new_triplets

    # 5. 获得 train_g 和 train_w 表示每个实体的邻居信息和邻居r权重分母
    def calcu_neighbor(self):
        train_g = np.ones((self.cnt_e, self.max_neighbor, 2), dtype=np.dtype('int64'))
        train_w = np.ones((self.cnt_e, self.max_neighbor), dtype=np.dtype('float32'))
        train_g[:, :, 0] *= self.cnt_r * 2  # 所有的0位置都是2r - 一个取不到的值
        train_g[:, :, 1] *= self.cnt_e  # 所有的0位置都是e - 一个取不到的值

        for ei in self.graph:
            rlist = self.graph[ei]  # list([rid,tid,weight], ...)
            num = min(self.max_neighbor, len(rlist))  # 保留的邻居信息数量
            train_g[ei][:num, :] = np.asarray(rlist)[:num, :2]  # (num, 2)
            train_w[ei][:num] = np.asarray(rlist)[:num, 2]  # (num, 1)
        return train_g, train_w

    # 4. 计算 Logic 权重分母: max({P(r'->r)|r' \in Nr(ei))})      graph:train+aux读取的数据
    def calcu_max_denominator(self):
        for e in self.graph:
            freq = defaultdict(int)  # 记录每个实体在rlist中出现的次数
            for r, t, value in self.graph[e]:
                freq[r] += 1
            if len(freq) == 1:  # 过滤了只有一条边的情况 other_rlist为空
                continue
            for ri in freq:
                other_rlist = [rel for rel in freq if ri != rel]
                imply_i = self.corr[ri]  # imply_i[j] == P(j->i)
                denominator = imply_i[other_rlist].max()
                for _id, neighbor in enumerate(self.graph[e]):
                    if neighbor[0] == ri:
                        self.graph[e][_id][2] = denominator
        return self.graph

    # 3. 计算 self.corr   graph:train读取的数据 不包括aux
    def calcu_correlation(self, graph):
        corr = np.zeros((self.cnt_r * 2 + 1, self.cnt_r * 2 + 1), dtype=np.dtype('float32'))
        freq = np.zeros((self.cnt_r * 2 + 1))  # 记录每个r出现的次数-分母
        for e in graph:
            neighbor_list = list(set([neighbor[0] for neighbor in graph[e]]))
            num_neighbor_list = len(neighbor_list)
            for id in range(num_neighbor_list):  # 遍历实体e的每个邻居信息 ri, ti
                ri = neighbor_list[id]
                freq[ri] += 1
                for id2 in range(id+1, num_neighbor_list):
                    rj = neighbor_list[id2]
                    corr[ri][rj] += 1
                    corr[rj][ri] += 1
        for ri in range(self.cnt_r * 2):
            corr[ri] = (corr[ri] * 1.0) / freq[ri]

        self.corr = corr.transpose()  #
        for ri in range(self.cnt_r * 2):
            corr[ri][ri] = corr[ri].mean()

    # 2. 获取cnt数据 + 构造graph(train+aux 并且用max_neighbor限制)
    def get_cnt_graph(self, train_triplets, aux_triplets, max_neighbor, args, logger):
        # 1. 获取 tr实体数量、tr+aux实体数量、关系数量
        h_set, r_set, t_set = [set([triple[id] for triple in train_triplets]) for id in range(3)]
        tr_e_set = h_set.union(t_set)  # train_entity_set
        self.cnt_tr_e = max(tr_e_set) + 1  # 加1是因为id是从0开始的 实际的个数=最大id+1 10336
        self.cnt_r = max(r_set) + 1  # 1170
        aux_h_set, aux_t_set = [set([triple[id] for triple in aux_triplets]) for id in [0, 2]]
        e_set = tr_e_set.union(aux_h_set.union(aux_t_set))
        self.cnt_e = max(e_set) + 1  # train + aux 的实体数量  14295

        # 2. 从 train_triplets 和 aux_triplets 和合法部分构造 graph
        graph = defaultdict(list)
        for h, r, t in train_triplets:
            graph[h].append([r, t, 0.])
            graph[t].append([r + self.cnt_r, t, 0.])

        # 3. 中途计算corr
        self.calcu_correlation(graph)  # 计算 correlation

        for h, r, t in aux_triplets:
            if r >= self.cnt_r:
                continue
            if h not in tr_e_set and t in tr_e_set:
                graph[h].append([r, t, 0.])
            if t not in tr_e_set and h in tr_e_set:
                graph[t].append([r + self.cnt_r, h, 0.])
        self.graph = graph

