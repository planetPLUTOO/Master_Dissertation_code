import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score

from basic.util import exclusive_combine


class Net2Net(nn.Module):

    def __init__(self, feature_num, label_num, AttGRU, encoder):
        super(Net2Net, self).__init__()
        self.AttGRU = AttGRU
        self.node_num = feature_num
        self.embed_dim = encoder.embed_dim * encoder.att_hop
        self.encoder = encoder
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim, label_num))
        init.xavier_uniform_(self.weight)


    def forward(self, nodes, labels):
        embeds = self.encoder(nodes)
        embeds = embeds.reshape(embeds.size(0), embeds.size(1)*embeds.size(2))
        scores = embeds.mm(self.weight)
        '''
        attT = att.transpose(1,2)
        identity = torch.eye(att.size(1))
        identity = Variable(identity.unsqueeze(0).expand(att.size(0),att.size(1),att.size(1))).cuda(device)
        penal = att @ attT - identity
        penal = torch.sum(torch.sum(torch.sum(penal ** 2, 1), 1) ** 0.5)
        '''

        return self.xent(scores, labels)


    def evaluate(self, labels, train_idx, test_idx):
        self.eval()
        embeds=self.encoder(test_idx)
        embeds = embeds.reshape(embeds.size(0), embeds.size(1)*embeds.size(2))
        scores = F.softmax(embeds.mm(self.weight))
        scores = scores.argmax(dim=1).cpu().numpy()
        acc = f1_score(labels[test_idx].cpu().numpy(), scores,average='micro')
        #acc=torch.eq(scores.argmax(dim=1),labels[test_idx]).float().sum()/1000
        '''
        hidden = []
        idx = []
        for bat in b_list:
            h = self.encoder(bat)
            h = h.reshape(h.size(0), h.size(1)*h.size(2))
            hidden.extend(h.detach().cpu().numpy())
            idx.extend(bat)
        hidden = np.stack(hidden,axis=0)
        hidden = hidden[np.argsort(idx)]
        lr = LogisticRegression(solver='liblinear',multi_class='auto')
        lr.fit(hidden[train_idx], labels[train_idx])
        '''
        return acc#lr.score(hidden[test_idx], labels[test_idx])


class MeanAggregator(nn.Module):
    def __init__(self, features, feat_dim, att_hop):

        super(MeanAggregator, self).__init__()
        self.a = nn.Parameter(torch.zeros(size=(att_hop, 2 * feat_dim)))
        init.xavier_uniform_(self.a.data, gain=1.414)
        self.features = features


    def forward(self, nodes, to_neighs):
        samp_neighs = [samp_neigh + [nodes[i]] for i, samp_neigh in enumerate(to_neighs)]

        unique_nodes_list = exclusive_combine(samp_neighs)
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        embed_matrix = self.features(unique_nodes_list)
        features_neighbor = [embed_matrix[[unique_nodes[neighbor] for neighbor in neighbors]] for neighbors in samp_neighs]
        features_concat = [torch.cat((features_neighbor[i],features_neighbor[i][-1].repeat(len(samp_neighs[i]),1,1)),dim=2) for i in range(len(samp_neighs))]

        att_graph = [F.leaky_relu(torch.sum(i*self.a, dim=2,keepdim=True)) for i in features_concat]
        att_graph = [F.softmax(i,dim=0) for i in att_graph]
        # att_graph = [F.softmax(F.leaky_relu(torch.sum(i*self.a, dim=2)),dim=0) for i in features_concat]
        att_features = [torch.sum(features_neighbor[i]*att_graph[i],dim=0) for i in range(len(features_neighbor))]
        att_features = torch.stack(att_features, dim=0)
        # to_feats = mask.mm(embed_matrix)
        return att_features  # node_num * fea_dim


class EgoEncoder(nn.Module):

    def __init__(self, feature_dim, embed_dim, att_hop, neighbor_dict, aggregator, base_model=None):
        super(EgoEncoder, self).__init__()

        self.feat_dim = feature_dim
        self.embed_dim = embed_dim
        self.att_hop = att_hop
        self.neighbor_dict = neighbor_dict
        self.aggregator = aggregator
        if base_model is not None:
            self.base_model = base_model

        self.weight = nn.Parameter(torch.FloatTensor(att_hop, feature_dim, embed_dim))
        init.xavier_uniform_(self.weight, gain=1.414)


    def forward(self, nodes):
        to_neighs = [self.neighbor_dict[node] for node in nodes]
        neigh_feats = self.aggregator.forward(nodes, to_neighs)
        combined = neigh_feats.transpose(1,0)
        combined = combined @ self.weight
        combined = torch.tanh(combined)
        return combined.transpose(1,0)  # node_num * emb_dim


class StructuredSelfAttention(torch.nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup
    """
    def __init__(self, content_code, batch_size, hid_dim, d_a, r, max_len, padding_idx, word_dim=500, vocab_size=None):

        super(StructuredSelfAttention, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, word_dim,padding_idx=padding_idx)
        self.content_code = content_code
        # self.doc_len = doc_len
        self.gru = nn.GRU(word_dim, hid_dim, 1, batch_first=True, bidirectional=True)
        self.linear_first = nn.Linear(hid_dim*2, d_a, bias=False)
        self.linear_second = nn.Linear(d_a, r, bias=False)
        self.batch_size = batch_size
        self.max_len = max_len
        self.hid_dim = hid_dim
        self.hidden_state = self.init_hidden(batch_size)
        self.r = r


    def init_hidden(self, num_nodes):
        return Variable(torch.zeros(2, num_nodes, self.hid_dim)).cuda()


    def forward(self, node_batch):
        # batch_doc_len = self.doc_len[node_batch]
        self.hidden_state = self.init_hidden(len(node_batch))

        # doc_sorted, doc_idx = torch.sort(batch_doc_len,descending=True)
        # node_batch = [node_batch[i] for i in doc_idx]
        node_content = self.content_code[node_batch]

        embeddings = self.embeddings(node_content)
        # embeddings_packed = pack_padded_sequence(embeddings, doc_sorted, batch_first=True)
        outputs, self.hidden_state = self.gru(embeddings, self.hidden_state)
        # outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        # outputs = outputs[torch.argsort(doc_idx)]

        x = torch.tanh(self.linear_first(outputs))
        x = self.linear_second(x)
        x = F.softmax(x, dim=1)
        attention = x.transpose(1, 2)
        sentence_embeddings = attention @ outputs
        # avg_sentence_embeddings = torch.sum(sentence_embeddings,1)/self.r
        # stack_sentence_embeddings = sentence_embeddings.view(sentence_embeddings.size(0), sentence_embeddings.size(1)*sentence_embeddings.size(2))
        return sentence_embeddings
