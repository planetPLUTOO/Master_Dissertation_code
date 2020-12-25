import numpy as np
import scipy.sparse as sp
import torch

from sklearn import metrics
from scipy.io import loadmat


def load_data(dataset):
    if dataset == 'yeast':
        adj = sp.csr_matrix(loadmat('../data/yeast/adjmat')['adjmat'])
        features = torch.FloatTensor(np.load('../data/yeast/features.npy'))
        labels = torch.FloatTensor(np.load('../data/yeast/labels.npy'))
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        idx_train = range(200)
        idx_val = range(200, 500)
        idx_test = range(500, 1000)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test


    if dataset == 'facebook':
        adj = sp.csr_matrix(np.load('../data/facebook/adj.npy'))
        features = torch.FloatTensor(np.load('../data/facebook/feature.npy'))
        labels = torch.FloatTensor(np.load('../data/facebook/label.npy'))
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        idx_train = range(100)
        idx_val = range(200,300)
        idx_test = range(300,450)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test

    if dataset == 'movie':
        adj = np.loadtxt('../data/movie/adj_dir.dat')
        adj[adj>1] = 1
        adj = sp.csr_matrix(adj)
        features = torch.FloatTensor(np.loadtxt('../data/movie/features(0-1).dat'))
        labels = torch.FloatTensor(np.loadtxt('../data/movie/labels.dat'))
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        idx_train = range(500)
        idx_val = range(500, 1000)
        idx_test = range(5000, 7000)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    output[output > 0.] = 1
    output[output <= 0.] = 0
    return metrics.f1_score(labels.detach().numpy(), output.detach().numpy(), average="micro")


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def init_sample_table(train_label):
    count = torch.sum(train_label, 0)
    pow_frequency = count**0.75
    power = sum(pow_frequency)
    ratio = pow_frequency / power
    table_size = 1e8
    count = np.round(ratio * table_size)
    sample_table = []
    for idx, x in enumerate(count):
        sample_table += [idx]*int(x)

    return np.array(sample_table)