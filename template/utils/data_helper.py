
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


class DataHelper:
    def __init__(self, args, logger):
        # 略 数据处理内容
        pass