from torch.utils.data import Dataset, DataLoader
from python.szt_logger import logFrame
import argparse
import time

# 文件信息
parser = argparse.ArgumentParser(description='TransE Data Process')
parser.add_argument('--prefix', type=str, default='..\\data\\freebase_mtr100_mte100-')
parser.add_argument('--type', type=str, default='train')
parser.add_argument('--log_path', type=str, default='..\\szt_loggings\\test\\{}_log.txt'.format(
    time.strftime("%Y_%m_%d %H_%M_%S", time.localtime())))
args = parser.parse_args()


args.path = args.prefix + args.type + '.txt'
args.eid_list_path = args.prefix + 'eid_list.txt'
args.rid_list_path = args.prefix + 'rid_list.txt'
print(args)

# logging
log = logFrame()
logger = log.getlogger(args.log_path)


# parser.add_argument('--batch', type=int, default=4)
# parser.add_argument('--load_model', action='store_true',
#                     help='Load existing model?')


# 返回 e_list, r_list
def create_er_id():
    e_list, r_list = [], []
    with open(args.path, 'r') as f:
        triples = f.readlines()
        for triple in triples:
            triple = triple.strip('\n')
            h, r, t = triple.split('\t')
            for entity in [h, t]:
                if entity not in e_list:
                    e_list.append(entity)
            if r not in r_list:
                r_list.append(r)
    with open(args.eid_list_path, 'w') as f_e:
        for index, entity in enumerate(e_list):
            f_e.write('%d %s\n' % (index, entity))
    with open(args.rid_list_path, 'w') as f_r:
        for index, relation in enumerate(r_list):
            f_r.write('%d %s\n' % (index, relation))
    return e_list, r_list


# 加载 path路径文件中 id和item之间的map
def load_id_map(path):
    id2item, item2id = {}, {}
    with open(path, 'r') as f:
        contents = f.readlines()
        for content in contents:
            index, item = content.strip('\n').split(' ')
            id2item[index] = item
            item2id[item] = int(index)
    return id2item, item2id


# 加载 id和entity/relation之间的map 加载id三元组
def load_triples():
    _id2e, _e2id = load_id_map(args.eid_list_path)
    _id2r, _r2id = load_id_map(args.rid_list_path)
    _id_triples = []
    with open(args.path, 'r') as f:
        triples = f.readlines()
        for triple in triples:
            h, r, t = triple.strip('\n').split('\t')
            h, t = _e2id[h], _e2id[t]
            r = _r2id[r]
            _id_triples.append([h, r, t])
    logger.info('加载数据：实体%d个，关系%d个，三元组%d个' %(len(_id2e), len(_id2r), len(_id_triples)))
    return _id2e, _e2id, _id2r, _r2id, _id_triples


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def test(self, index):
        return self.data[index]

# create_er_id()  # 根据三元组信息 生成entity和relation的列表并存储




