from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.optim as optim
import itertools

from pygcn.utils import load_data, accuracy, init_sample_table
from pygcn.models import GCN, MLP, LE


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='yeast',
                    help='load dataset.')
parser.add_argument('--model', type=str, default='GCN',
                    help='MLP or GCN.')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--sample_num', type=int, default=5,
                    help='Number of negative sample.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=args.dataset)
label_layer = LE(labels.shape[1], args.hidden)
sample_table = init_sample_table(labels[idx_train])

# Model and optimizer
if args.model=='GCN':
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.shape[1],
                dropout=args.dropout)

elif args.model=='MLP':
    model = MLP(nfeat=adj.shape[1],
                nhid=args.hidden,
                nclass=labels.shape[1],
                dropout=args.dropout)


optimizer = optim.Adam(itertools.chain(model.parameters(), label_layer.parameters()),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    hidden, output = model(features, adj)
    label_embedding = label_layer()
    sum_loss = 0
    sum_label_loss = 0
    node_with_no_labels = 0
    less_than_two_labels = 0

    for i in idx_train:
        label_index = np.where(labels[i])[0]
        if(len(label_index)==0):
            node_with_no_labels += 1
            less_than_two_labels +=1
            continue
        label_loss = 0

        for j in label_index:
            sample_index1 = np.random.choice(sample_table, args.sample_num)
            while j in sample_index1:
                idx = np.where(sample_index1==j)[0]
                resample = np.random.choice(sample_table, len(idx))
                sample_index1 = np.delete(sample_index1, idx)
                sample_index1 = np.append(sample_index1, resample)

            sum_loss += (-torch.log(torch.sigmoid(torch.sum(hidden[i] * label_embedding[j]))) - torch.sum(torch.log(
                        torch.sigmoid(torch.sum(-hidden[i] * label_embedding[sample_index1], 1)))))/ len(label_index)
            
            if len(label_index)==1:
                less_than_two_labels +=1
                continue

            for k in label_index:
                if j==k:
                    continue
                sample_index2 = np.random.choice(sample_table, args.sample_num)
                while k in sample_index2:
                    idx = np.where(sample_index2 == k)[0]
                    resample = np.random.choice(sample_table, len(idx))
                    sample_index2 = np.delete(sample_index2, idx)
                    sample_index2 = np.append(sample_index2, resample)

                label_loss += (-torch.log(torch.sigmoid(torch.sum(label_embedding[j] * label_embedding[k]))) - torch.sum(torch.log(
                        torch.sigmoid(torch.sum(-label_embedding[j] * label_embedding[sample_index2], 1))))) / (len(label_index)-1)

        sum_label_loss += (label_loss / len(label_index))

    sum_loss /= (len(idx_train) - node_with_no_labels)
    sum_label_loss /= (len(idx_train)-less_than_two_labels)

    loss_train = torch.nn.BCEWithLogitsLoss()(output[idx_train], labels[idx_train])
    loss_sum = loss_train + 0.25*sum_loss + 0.25*sum_label_loss
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_sum.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        hidden, output = model(features, adj)
    
    loss_val = torch.nn.BCEWithLogitsLoss()(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    hidden, output = model(features, adj)
    loss_test = torch.nn.BCEWithLogitsLoss()(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
