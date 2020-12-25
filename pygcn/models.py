import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution, Dense


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass, bias=False)
        self.dropout = dropout

    def forward(self, x, adj):
        hidden = F.relu(self.gc1(x, adj))
        x = F.dropout(hidden, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return hidden, x


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()

        self.dense1 = Dense(nfeat, nhid)
        self.dense2 = Dense(nhid, nclass, bias=False)
        self.dropout = dropout

    def forward(self, x):
        hidden = F.relu(self.dense1(x))
        x = F.dropout(hidden, self.dropout, training=self.training)
        x = self.dense2(x)
        return hidden, x

class LE(nn.Module):
    def __init__(self, nlabels, label_dim):
        super(LE, self).__init__()

        self.label_embedding = nn.Parameter(torch.FloatTensor(nlabels, label_dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.label_embedding.size(1))
        self.label_embedding.data.uniform_(-stdv, stdv)

    def forward(self):
        return self.label_embedding
