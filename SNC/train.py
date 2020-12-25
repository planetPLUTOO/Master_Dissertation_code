import itertools

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, average_precision_score, roc_curve

from utils import batch_separator, negative_sample


def train(edges, edge_dict_u, edge_dict_v, u_idx_dict, v_idx_dict, data_train, data_test,
          cnn_u, cnn_v, sample_table_u, sample_table_v, args):
    '''
    roc_best = 0
    pr_best = 0
    count = 0
    '''
    optimizer = torch.optim.SGD(itertools.chain(cnn_u.parameters(), cnn_v.parameters()), lr=args.lr,
                                weight_decay=0.001)

    lg = LogisticRegression(solver='liblinear', penalty='l2', C=0.001)

    batches = batch_separator(edges, args.batch_size)
    for epoch in range(1, args.epochs+1):
        cnn_u.train()
        cnn_v.train()

        for num, batch in enumerate(batches):
            optimizer.zero_grad()
            node_u = [i[0] for i in batch]
            node_v = [i[1] for i in batch]

            if args.loss_type == 'hinge_loss':
                idx_v_dict = dict(zip(v_idx_dict.values(), v_idx_dict.keys()))
                idx_u_dict = dict(zip(u_idx_dict.values(), u_idx_dict.keys()))
                u_sample, u_dict_with_neg = negative_sample(node_v, node_u, idx_u_dict, sample_table_u, args.sample_num)
                v_sample, v_dict_with_neg = negative_sample(node_u, node_v, idx_v_dict, sample_table_v, args.sample_num)

                u_embed = cnn_u(list(u_dict_with_neg.keys()), edge_dict_u, v_idx_dict)
                v_embed = cnn_v(list(v_dict_with_neg.keys()), edge_dict_v, u_idx_dict)

                pos_u_embed = u_embed[[u_dict_with_neg[i] for i in node_u]]
                pos_v_embed = v_embed[[v_dict_with_neg[i] for i in node_v]]

                loss = 0
                for i in range(args.sample_num):
                    neg_u = [j[i] for j in u_sample]
                    neg_u_embed = u_embed[[u_dict_with_neg[j] for j in neg_u]]
                    loss += F.pairwise_distance(pos_u_embed, pos_v_embed)**2 - F.pairwise_distance(pos_v_embed, neg_u_embed)**2 + 2.0
                loss[loss<0] = 0
                for j in range(args.sample_num):
                    neg_v = [j[i] for j in v_sample]
                    neg_v_embed = v_embed[[v_dict_with_neg[j] for j in neg_v]]
                    loss += F.pairwise_distance(pos_u_embed, pos_v_embed) ** 2 - F.pairwise_distance(pos_u_embed, neg_v_embed)**2 + 2.0
                loss[loss<0] = 0

                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()

            elif args.loss_type == '2nd_order' or args.loss_type == '1st+2nd':
                idx_v_dict = dict(zip(v_idx_dict.values(), v_idx_dict.keys()))
                v_sample, v_dict_with_neg = negative_sample(node_u, node_v, idx_v_dict, sample_table_v, args.sample_num)

                u_embed = cnn_u(node_u, edge_dict_u, v_idx_dict)
                v_embed = cnn_v(list(v_dict_with_neg.keys()), edge_dict_v, u_idx_dict)

                pos_v_embed = v_embed[[v_dict_with_neg[i] for i in node_v]]
                loss = -torch.log(torch.sigmoid(torch.sum(u_embed*pos_v_embed, 1)))
                for i in range(args.sample_num):
                    neg_v = [j[i] for j in v_sample]
                    neg_v_embed = v_embed[[v_dict_with_neg[j] for j in neg_v]]
                    loss -= torch.log(torch.sigmoid(-torch.sum(neg_v_embed*u_embed, 1)))
                loss = torch.mean(loss)

                loss.backward()
                optimizer.step()

            else:
                u_embed = cnn_u(node_u, edge_dict_u, v_idx_dict)
                v_embed = cnn_v(node_v, edge_dict_v, u_idx_dict)

                loss = torch.sigmoid(torch.sum(u_embed*v_embed, 1))
                loss = - torch.mean(torch.log(loss))
                loss.backward()
                optimizer.step()

            if num+1 == len(batches):
                print('\repoch: %d batch: [%d/%d] complete, loss: %0.4f' % (epoch, num+1, len(batches), loss), end='\n')
            else:
                print('\repoch: %d batch: [%d/%d] complete, loss: %0.4f' % (epoch, num+1, len(batches), loss), end='')

        if epoch%50 == 0:
            cnn_u.eval()
            cnn_v.eval()

            if args.dataset == 'wiki':
                u_train_embed = np.zeros([15000,args.embed_size])
                u_train_embed[0:7500] = cnn_u(list(u_idx_dict.keys())[:7500], edge_dict_u, v_idx_dict).detach().cpu().numpy()
                u_train_embed[7500:] = cnn_u(list(u_idx_dict.keys())[7500:], edge_dict_u, v_idx_dict).detach().cpu().numpy()

                v_train_embed = cnn_v(list(v_idx_dict.keys()), edge_dict_v, u_idx_dict).detach().cpu().numpy()

            elif args.dataset == 'DBLP':
                u_train_embed = np.zeros([len(u_idx_dict), args.embed_size])
                u_train_embed[0:int(len(u_idx_dict)/2)] = cnn_u(list(u_idx_dict.keys())[:int(len(u_idx_dict)/2)], edge_dict_u,
                                                           v_idx_dict).detach().cpu().numpy()
                u_train_embed[int(len(u_idx_dict)/2):] = cnn_u(list(u_idx_dict.keys())[int(len(u_idx_dict)/2):], edge_dict_u,
                                                          v_idx_dict).detach().cpu().numpy()
                v_train_embed = cnn_v(list(v_idx_dict.keys()), edge_dict_v, u_idx_dict).detach().cpu().numpy()

            elif args.dataset == 'movielen':
                u_train_embed = cnn_u(list(u_idx_dict.keys()), edge_dict_u,v_idx_dict).detach().cpu().numpy()
                v_train_embed = cnn_v(list(v_idx_dict.keys()), edge_dict_v, u_idx_dict).detach().cpu().numpy()

            #np.save('u_feature', u_train_embed)
            #np.save('v_feature', v_train_embed)

            if args.loss_type == '1st+2nd':
                u_feature_1st = np.load('u_feature.npy')
                v_feature_1st = np.load('v_feature.npy')
                u_train_embed = np.hstack([u_train_embed, u_feature_1st])
                v_train_embed = np.hstack([v_train_embed, v_feature_1st])
            
            train_feature = np.zeros([len(data_train), args.embed_size*4])
            for num, train_pair in enumerate(data_train):
                if train_pair[0] in u_idx_dict.keys():
                    train_feature[num, :args.embed_size*2] = u_train_embed[u_idx_dict[train_pair[0]]]
                if train_pair[1] in v_idx_dict.keys():
                    train_feature[num, args.embed_size*2:] = v_train_embed[v_idx_dict[train_pair[1]]]

            train_label = np.array([i[2] for i in data_train]).astype('int')
            lg.fit(train_feature, train_label)

            test_feature = np.zeros([len(data_test), args.embed_size*4])
            for num, test_pair in enumerate(data_test):
                if test_pair[0] in u_idx_dict.keys():
                    test_feature[num, :args.embed_size*2] = u_train_embed[u_idx_dict[test_pair[0]]]
                if test_pair[1] in v_idx_dict.keys():
                    test_feature[num, args.embed_size*2:] = v_train_embed[v_idx_dict[test_pair[1]]]

            y_pre = lg.predict_proba(test_feature)[:, 1]
            test_labels = np.array([i[2] for i in data_test]).astype('int')
            fpr, tpr, thresholds = roc_curve(test_labels, y_pre)
            average_precision = average_precision_score(test_labels, y_pre)
            AUC_ROC = auc(fpr, tpr)
            '''
            if AUC_ROC > roc_best:
                np.save('u_feature', u_feature)
                np.save('v_feature', v_feature)
                roc_best = AUC_ROC

            if AUC_ROC < roc_best and average_precision < pr_best:
                count += 1
                if count > args.early_stop:
                    print('early stop, epoch: %d  AUC_ROC: %0.4f  AUC_PR:  %0.4f' % (epoch, AUC_ROC, average_precision))
                    break
            else:
                count = 0
                '''
            print('epoch: %d  AUC_ROC: %0.4f  AUC_PR:  %0.4f' %(epoch, AUC_ROC, average_precision))
