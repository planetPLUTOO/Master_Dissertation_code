# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import feature_constructor


class CNN_u(nn.Module):

    def __init__(self, edge_dict_v, args):
        super(CNN_u, self).__init__()
        self.args = args

        self.v_feature = torch.rand(len(edge_dict_v), args.feature_size)

        D_v = args.feature_size
        Ci = 1  # channel num
        Co = args.kernel_num  # number of one kernel
        K_v = args.kernel_sizes_v  # size of kernels

        self.v_convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D_v)) for K in K_v])  # nn.conv2d(in_channel, out_channel, kernel_size)

        self.BN = nn.BatchNorm1d(args.embed_size)

        self.v_fc = nn.Linear(len(K_v)*Co, args.embed_size)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)

    def forward(self, node_u, edge_dict_u, v_idx_dict):
        u_neighbor = [edge_dict_u[i] for i in node_u]

        batch_feature = feature_constructor(self.args, self.v_feature, u_neighbor, v_idx_dict)  # [batch_size, 1, max_degree, D_author]

        x = [conv(batch_feature).squeeze(3) for conv in self.v_convs]  # [batch_size, kernel_num, max_degree-K+1, 1]

        x = [F.leaky_relu(i) for i in x]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [batch_size, kernel_num]

        x = F.leaky_relu(torch.cat(x, 1))  # [batch_size, kernel_num*len(Ka)]

        u_embed = self.BN(self.v_fc(x))  # [batch_size, embed_size]

        return u_embed


class CNN_v(nn.Module): # convolute paper feature, getting the author embedding

    def __init__(self, edge_dict_u, args):
        super(CNN_v, self).__init__()
        self.args = args

        self.u_feature = torch.rand(len(edge_dict_u), args.feature_size)

        D_u = args.feature_size
        Ci = 1
        Co = args.kernel_num
        K_u = args.kernel_sizes_u

        self.u_convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D_u)) for K in K_u])

        self.BN = nn.BatchNorm1d(args.embed_size)

        self.u_fc = nn.Linear(len(K_u)*Co, args.embed_size)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)

    def forward(self, node_v, edge_dict_v, u_idx_dict):
        v_neighbor = [edge_dict_v[i] for i in node_v]

        batch_feature = feature_constructor(self.args, self.u_feature, v_neighbor, u_idx_dict)

        x = [conv(batch_feature).squeeze(3) for conv in self.u_convs]  # [batch_size, kernel_num, max_degree-K+1, 1]

        x = [F.leaky_relu(i) for i in x]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [batch_size, kernel_num]

        x = F.leaky_relu(torch.cat(x, 1))  # [batch_size, kernel_num*len(Ka)]

        v_embed = self.BN(self.u_fc(x))

        return v_embed