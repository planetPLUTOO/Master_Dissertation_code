import numpy as np
import torch
import os


def load_data(dataset):
    if dataset=='wiki':
        filename = os.path.join('data', 'wiki', "rating_train.dat")
        edge_dict_u = {}
        edge_dict_v = {}
        edges = []
        with open(filename, encoding="UTF-8") as fin:
            line = fin.readline()
            while line:
                user, item, rating = line.strip().split("\t")
                edges.append([user, item])
                if edge_dict_u.get(user) is None:
                    edge_dict_u[user] = []
                if edge_dict_v.get(item) is None:
                    edge_dict_v[item] = []
                edge_dict_u[user].append(item)
                edge_dict_v[item].append(user)
                line = fin.readline()
        node_u = dict(zip(list(edge_dict_u.keys()), [i for i in range(len(edge_dict_u))]))
        node_v = dict(zip(list(edge_dict_v.keys()), [i for i in range(len(edge_dict_v))]))

        return edges, edge_dict_u, edge_dict_v, node_u, node_v

    elif dataset == 'DBLP':
        filename = os.path.join('data', 'DBLP', "edge_train.txt")
        edge_dict_u = {}
        edge_dict_v = {}
        edges = []
        with open(filename, encoding="UTF-8") as fin:
            line = fin.readline()
            while line:
                user, item, rating = line.strip().split("\t")
                edges.append([user, item])
                if edge_dict_u.get(user) is None:
                    edge_dict_u[user] = []
                if edge_dict_v.get(item) is None:
                    edge_dict_v[item] = []
                edge_dict_u[user].append(item)
                edge_dict_v[item].append(user)
                line = fin.readline()
        node_u = dict(zip(list(edge_dict_u.keys()), [i for i in range(len(edge_dict_u))]))
        node_v = dict(zip(list(edge_dict_v.keys()), [i for i in range(len(edge_dict_v))]))

        return edges, edge_dict_u, edge_dict_v, node_u, node_v

    elif dataset == 'movielen':
        filename = os.path.join('data', 'movielen', "edge_train.txt")
        edge_dict_u = {}
        edge_dict_v = {}
        edges = []
        with open(filename, encoding="UTF-8") as fin:
            line = fin.readline()
            while line:
                user, item, rating = line.strip().split("\t")
                edges.append([user, item])
                if edge_dict_u.get(user) is None:
                    edge_dict_u[user] = []
                if edge_dict_v.get(item) is None:
                    edge_dict_v[item] = []
                edge_dict_u[user].append(item)
                edge_dict_v[item].append(user)
                line = fin.readline()
        node_u = dict(zip(list(edge_dict_u.keys()), [i for i in range(len(edge_dict_u))]))
        node_v = dict(zip(list(edge_dict_v.keys()), [i for i in range(len(edge_dict_v))]))

        return edges, edge_dict_u, edge_dict_v, node_u, node_v