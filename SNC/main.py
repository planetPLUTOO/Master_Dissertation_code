import argparse
import numpy as np
import model
import train

from utils import init_sample_table, train_and_test_data
from load_data import load_data


parser = argparse.ArgumentParser(description='CNN text classificer')
# data
parser.add_argument('-dataset', type=str, default='movielen', help='choose the dataset from [wiki, DBLP, movielen]')
parser.add_argument('-feature_size', type=int, default=128, help='feature size of each node')
# learning
parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate [default: 0.01]')
parser.add_argument('-epochs', type=int, default=50, help='number of epochs for train [default: 256]')
# parser.add_argument('-early_stop', type=int, default=5, help='iteration numbers to stop without performance increasing')
# parser.add_argument('-save_best', type=bool, default=True, help='whether to save when get best performance')
parser.add_argument('-batch_size', type=int, default=1000, help='batch size of learning')
# model
parser.add_argument('-n_neighbors', type=int, default=20)
parser.add_argument('-kernel_num', type=int, default=64, help='number of each kind of kernel')
parser.add_argument('-kernel_sizes_u', type=list, default=[2,3,4,5], help='comma-separated kernel size to use for convolution')
parser.add_argument('-kernel_sizes_v', type=list, default=[2,3,4,5], help='comma-separated kernel size to use for convolution')
parser.add_argument('-embed_size', type=int, default=64, help='dim of embedding vectors')
parser.add_argument('-sample_num', type=int, default=10, help='number of node when negative sample')
parser.add_argument('-loss_type', type=str, default='1st+2nd', help='1st_order, 2nd_order, 1st+2nd or hinge_loss')
# device
parser.add_argument('-cuda', type=bool, default=True, help='disable the gpu')

args = parser.parse_args()

np.random.seed(123)
# load data
edges, edge_dict_u, edge_dict_v, u_idx_dict, v_idx_dict = load_data(dataset=args.dataset)
data_train, data_test = train_and_test_data(dataset=args.dataset)
sample_table_u, sample_table_v = init_sample_table(edge_dict_u, edge_dict_v)

# model
cnn_u = model.CNN_u(edge_dict_v, args)
cnn_v = model.CNN_v(edge_dict_u, args)
if args.cuda:
    cnn_u = cnn_u.cuda()
    cnn_v = cnn_v.cuda()

train.train(edges, edge_dict_u, edge_dict_v, u_idx_dict, v_idx_dict, data_train, data_test,
            cnn_u, cnn_v, sample_table_u, sample_table_v, args)