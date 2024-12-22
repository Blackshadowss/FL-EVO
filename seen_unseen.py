# from rgcn.utils import load_data, build_sub_graph
import numpy as np
import torch
import dgl
from collections import defaultdict
import pickle
import os
import knowledge_graph as knwlgrh


def load_data(dataset, bfs_level=3, relabel=False):
    if dataset in ['aifb', 'mutag', 'bgs', 'am']:
        return knwlgrh.load_entity(dataset, bfs_level, relabel)
    elif dataset in ['FB15k', 'wn18', 'FB15k-237']:
        return knwlgrh.load_link(dataset)
    elif dataset in ['ICEWS18', 'ICEWS14', "GDELT", "SMALL", "ICEWS14s", "ICEWS05-15", "YAGO",
                     "WIKI"]:
        print(os.getcwd())
        return knwlgrh.load_from_local("../data", dataset)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def split_with_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    snapshot_time = [latest_t]
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        # latest_t表示读取的上一个三元组发生的时刻，要求数据集中的三元组是按照时间发生顺序排序的
        if latest_t != t:  # 同一时刻发生的三元组
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
                snapshot_time.append(latest_t)
            snapshot = []
        snapshot.append(train[:3])
    # 加入最后一个shapshot
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:, 0], snapshot[:, 2]), return_inverse=True)  # relabel
        uniq_r = np.unique(snapshot[:, 1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r) * 2)
    print(
        "# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
        .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list),
                max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num),
                min(union_num)))
    return (snapshot_list, snapshot_time)


import pickle


def save_hist(filepath, snap_hist):
    f_save = open(filepath, 'wb')
    pickle.dump(snap_hist, f_save)
    f_save.close()


def seen_entity(num_nodes, num_rels, data):
    entity_occ = {}
    for entity_id in range(0, num_nodes):
        if entity_id not in entity_occ.keys():
            entity_occ[entity_id] = set()
        for timid, time in enumerate(data):
            if entity_id in time[:, 0] or entity_id in time[:, 2]:
                entity_occ[entity_id].add(timid)
                continue
    return entity_occ


def _read_triplets(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line  # 迭代执行和return直接停止不同


def _read_triplets_as_list(filename, entity_dict, relation_dict, load_time):
    l = []
    for triplet in _read_triplets(filename):
        s = int(triplet[0])
        r = int(triplet[1])
        o = int(triplet[2])
        if load_time:
            st = int(triplet[3])
            # et = int(triplet[4])
            # l.append([s, r, o, st, et])
            l.append([s, r, o, st])
        else:
            l.append([s, r, o])
    return l


def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            d[int(line[1])] = line[0]
    return d


def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        # latest_t表示读取的上一个三元组发生的时刻，要求数据集中的三元组是按照时间发生顺序排序的
        if latest_t != t:  # 同一时刻发生的三元组
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:4])  # Liu 20230619加入时间
    # 加入最后一个shapshot
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:, 0], snapshot[:, 2]), return_inverse=True)  # relabel
        uniq_r = np.unique(snapshot[:, 1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r) * 2)
    return snapshot_list


def unseen_seen_data(data_name, start_time, time_interval, seen_unseen, rule_dir_num):
    # load data
    data = load_data(data_name)
    train = split_with_time(data.train)
    dev = split_with_time(data.valid)
    test = split_with_time(data.test)
    num_rels = data.num_rels
    num_nodes = data.num_nodes

    train_entity = seen_entity(num_nodes, num_rels, train[0])
    test_entity = seen_entity(num_nodes, num_rels, train[0] + dev[0] + test[0])

    # unseen data

    unseen_entity = list()
    if seen_unseen == "unseen":
        for i in train_entity.keys():
            if len(train_entity[i]) == 0:
                unseen_entity.append(i)
    else:
        for i in train_entity.keys():
            if len(train_entity[i]) != 0:
                unseen_entity.append(i)

    unseen_triple = dict()
    unseen_triple_idx = dict()
    time = start_time
    for timeid, data in enumerate(test[0]):
        if time not in unseen_triple.keys():
            unseen_triple[time] = list()
            unseen_triple_idx[time] = list()
        for idx, each_np in enumerate(data):
            if seen_unseen == "unseen":
                if each_np[0] in unseen_entity or each_np[2] in unseen_entity:
                    unseen_triple[time].append(each_np)
                    unseen_triple_idx[time].append(idx)
            else:
                if each_np[0] in unseen_entity and each_np[2] in unseen_entity:
                    unseen_triple[time].append(each_np)
                    unseen_triple_idx[time].append(idx)
        time = time + time_interval

    with open('../data/{}/{}_test.txt'.format(data_name, seen_unseen), 'w', encoding='utf-8') as f:
        for key in unseen_triple.keys():
            unseen_data = [i.tolist() + [key, -1] for i in unseen_triple[key]]
            for v in unseen_data:
                f.write("\t".join([str(i) for i in v]))
                f.write("\n")

    dir = "../data/{}".format(data_name)
    entity_path = os.path.join(dir, 'entity2id.txt')
    relation_path = os.path.join(dir, 'relation2id.txt')
    entity_dict = _read_dictionary(entity_path)
    relation_dict = _read_dictionary(relation_path)
    test_unseen_path = os.path.join(dir, "{}_test.txt".format(seen_unseen))
    test_unseen = np.array(_read_triplets_as_list(test_unseen_path, entity_dict, relation_dict, load_time=True))
    test_unseen = split_by_time(test_unseen)

    save_hist("../data/{}/{}_test.pkl".format(data_name, seen_unseen), test_unseen)

    time = start_time
    for i in range(0, rule_dir_num):
        print(i)
        test_rule_total = pickle.load(open("../data/{}/test/".format(data_name) + str(i) + ".pkl", 'rb'))
        temp = test_rule_total[unseen_triple_idx[time]]
        save_hist("../data/{}/test_{}/".format(data_name, seen_unseen) + str(i) + ".pkl", temp)
        time = time + time_interval

def unseen_entity(data_name, seen_unseen):
    # load data
    data = load_data(data_name)
    train = split_with_time(data.train)
    dev = split_with_time(data.valid)
    test = split_with_time(data.test)
    num_rels = data.num_rels
    num_nodes = data.num_nodes

    train_entity = seen_entity(num_nodes, num_rels, train[0])
    test_entity = seen_entity(num_nodes, num_rels, train[0] + dev[0] + test[0])

    # unseen data

    unseen_entity = set()
    if seen_unseen == "unseen":
        for i in train_entity.keys():
            if len(train_entity[i]) == 0:
                unseen_entity.add(i)
    else:
        for i in train_entity.keys():
            if len(train_entity[i]) != 0:
                unseen_entity.add(i)
    print(data_name, len(unseen_entity))




#unseen_seen_data("ICEWS14", 7536, 24, "seen", 51)
#unseen_seen_data("ICEWS14", 7536, 24, "unseen", 51)
unseen_entity("ICEWS14","unseen")
unseen_entity("ICEWS18","unseen")