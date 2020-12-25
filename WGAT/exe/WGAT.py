import time

import numpy as np
import torch

from basic.util import padding, adj_list, get_batches, read_word_code
from model.models import MeanAggregator, EgoEncoder, StructuredSelfAttention, Net2Net



def main():
    start = time.time()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)


    data_dir = '../data_full_text/Citeseer/'
    adj_file = 'edges.txt'
    label_file = 'labels.txt'
    con_file = 'title.txt'
    vocab_file = 'voc.txt'
    vocab_size = 5523
    max_len = 30
    idx = np.arange(4610)
    np.random.shuffle(idx)
    train_idx = idx[:461]
    test_idx = idx[3610:4610]

    '''
    data_dir = '../data_full_text/Cora/'
    adj_file = 'edges.txt'
    label_file = 'labels.txt'
    con_file = 'abstract.txt'
    vocab_file = 'voc.txt'
    vocab_size = 12619
    max_len = 100
    idx = np.arange(1990)
    np.random.shuffle(idx)
    train_idx = idx[:199]
    test_idx = idx[990:1990]

    
    data_dir = '../data_full_text/DBLP/'
    adj_file = 'edges.txt'
    label_file = 'labels.txt'
    con_file = 'title.txt'
    vocab_file = 'voc.txt'
    vocab_size = 8501
    max_len = 30
    idx = np.arange(8501)
    np.random.shuffle(idx)
    train_idx = idx[:850]
    test_idx = idx[7501:8501]
    '''

    word_dim = 500
    hid_dim = 100
    enc_dim_first = 100
    enc_dim_second = 8
    att_hops = 4
    # batch_size = 32
    epoch_num = 100
    l_rate = 1e-3
    d_a = 100

    labels = torch.LongTensor(np.loadtxt(data_dir + label_file))[:, 1].cuda()
    edges = torch.LongTensor(np.loadtxt(data_dir + adj_file))
    node_content, pad_code = read_word_code(data_dir + con_file, data_dir + vocab_file, max_len)
    node_content = torch.LongTensor(node_content).cuda()
    feature_num = len(labels)
    label_num = torch.max(labels)+1
    # features = torch.LongTensor(np.loadtxt(data_dir + feature_file))
    # labels = torch.LongTensor(np.loadtxt(data_dir + label_file))[:,1].cuda()
    # edges = torch.LongTensor(np.loadtxt(data_dir + edge_file))

    neighbor_dict = adj_list(edges)
    # vocab_size = features.shape[1]
    # node_content = padding(features, max_len, vocab_size)
    # doc_len = torch.sum(features, dim=1)

    AttGRU = StructuredSelfAttention(node_content,batch_size=batch_size,hid_dim=hid_dim,d_a=d_a,r=att_hops,
                                     max_len=max_len,padding_idx=pad_code,word_dim=word_dim,vocab_size=vocab_size)

    agg1 = MeanAggregator(lambda nodes: AttGRU(nodes), hid_dim*2, att_hops)
    enc1 = EgoEncoder(hid_dim*2, enc_dim_first, att_hops, neighbor_dict, agg1)

    agg2 = MeanAggregator(lambda nodes: enc1(nodes),enc_dim_first, att_hops)
    enc2 = EgoEncoder(enc_dim_first, enc_dim_second, att_hops, neighbor_dict, agg2, base_model=enc1)

    c2n = Net2Net(feature_num, label_num, AttGRU, enc2)
    c2n.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, c2n.parameters()), lr=l_rate)

    for e in range(epoch_num):
        avg_loss = []
        c2n.train()
        # batch_list = get_batches(len(neighbor_dict), batch_size)
        batch = list(train_idx)
        #for batch in batch_list:
        optimizer.zero_grad()
        loss = c2n(batch, labels[batch])
        loss.backward()
        optimizer.step()
        #avg_loss.append(loss.item())

        # node classification results
        f1_micro = c2n.evaluate(labels, train_idx, test_idx).item()
        minute = np.around((time.time() - start) / 60)
        print('Epoch:', e, 'loss:', loss, 'mi-F1:', f1_micro, 'time:', minute, 'mins.')
        # ls = np.mean(avg_loss)
        #print('Epoch:', e, 'loss:', loss, 'mi-F1:', np.around(f1_micro, 3), 'time:', minute, 'mins.')
        # avg_loss.clear()

if __name__ == "__main__":
    main()
