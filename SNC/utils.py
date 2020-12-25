# coding:utf-8
import torch
import os
import numpy as np

def batch_separator(edges, batch_size):
    np.random.shuffle(edges)
    batches = [edges[i:i+batch_size] for i in range(0, len(edges), batch_size)]

    return batches


def feature_constructor(args, features, neighbor_nodes, dict):
    batch_feature = torch.zeros(len(neighbor_nodes), args.n_neighbors, args.feature_size)
    if args.cuda:
        batch_feature = batch_feature.cuda()

    for num, node in enumerate(neighbor_nodes):
        while len(node) < args.n_neighbors:
            node += node

        np.random.shuffle(node)
        node = [node[i] for i in range(args.n_neighbors)]
        node_idx = [dict[i] for i in node]
        feature = features[node_idx]
        batch_feature[num] = feature

    batch_feature = batch_feature.unsqueeze(1)  # add the channel dimention

    return batch_feature


def train_and_test_data(dataset="wiki"):
    if dataset == 'wiki':
        train_path = os.path.join('data', 'wiki', 'case_train.dat')
        test_path = os.path.join('data', 'wiki', 'case_test.dat')

        data_train = []
        data_test = []

        with open(train_path, encoding="UTF-8") as fin:
            line = fin.readline()
            while line:
                user, item, rating = line.strip().split("\t")
                data_train.append([user, item, rating])
                line = fin.readline()

        with open(test_path, encoding="UTF-8") as fin:
            line = fin.readline()
            while line:
                user, item, rating = line.strip().split("\t")
                data_test.append([user, item, rating])
                line = fin.readline()

        return data_train, data_test

    elif dataset == 'DBLP':
        train_path = os.path.join('data', 'DBLP', 'data_train.txt')
        test_path = os.path.join('data', 'DBLP', 'data_test.txt')

        data_train = []
        data_test = []

        with open(train_path, encoding="UTF-8") as fin:
            line = fin.readline()
            while line:
                user, item, rating = line.strip().split("\t")
                data_train.append([user, item, rating])
                line = fin.readline()

        with open(test_path, encoding="UTF-8") as fin:
            line = fin.readline()
            while line:
                user, item, rating = line.strip().split("\t")
                data_test.append([user, item, rating])
                line = fin.readline()

        return data_train, data_test

    elif dataset == 'movielen':
        train_path = os.path.join('data', 'movielen', 'data_train.txt')
        test_path = os.path.join('data', 'movielen', 'data_test.txt')

        data_train = []
        data_test = []

        with open(train_path, encoding="UTF-8") as fin:
            line = fin.readline()
            while line:
                user, item, rating = line.strip().split("\t")
                data_train.append([user, item, rating])
                line = fin.readline()

        with open(test_path, encoding="UTF-8") as fin:
            line = fin.readline()
            while line:
                user, item, rating = line.strip().split("\t")
                data_test.append([user, item, rating])
                line = fin.readline()

        return data_train, data_test

def init_sample_table(edge_dict_u, edge_dict_v):
    u_count = []
    for i in edge_dict_u.values():
        u_count.append(len(i))

    pow_frequency = [i**0.75 for i in u_count]
    power = sum(pow_frequency)
    ratio = [i / power for i in pow_frequency]
    table_size = 1e5
    count = np.round([i * table_size for i in ratio])
    sample_table_u = []
    for idx, x in enumerate(count):
        sample_table_u += [idx]*int(x)

    v_count = []
    for i in edge_dict_v.values():
        v_count.append(len(i))

    pow_frequency = [i**0.75 for i in v_count]
    power = sum(pow_frequency)
    ratio = [i / power for i in pow_frequency]
    table_size = 1e5
    count = np.round([i * table_size for i in ratio])
    sample_table_v = []
    for idx, x in enumerate(count):
        sample_table_v += [idx]*int(x)

    return np.array(sample_table_u), np.array(sample_table_v)

def negative_sample(node, pos_node, idx_node_dict, sample_table, K):
    sample_idx = np.random.choice(sample_table, len(node)*K)
    sample_node = [idx_node_dict[i] for i in sample_idx]
    pos_and_neg = list(set(pos_node + sample_node))
    pos_and_neg = dict(zip(pos_and_neg, [i for i in range(len(pos_and_neg))]))
    sample_node = [sample_node[i:i+K] for i in range(0, len(sample_node), K)]

    return sample_node, pos_and_neg