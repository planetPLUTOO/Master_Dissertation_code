import numpy as np
import torch


def read_word_code(text_path, vocab_path, max_len):
    words = []
    fin = open(vocab_path, 'r', encoding='UTF-8')
    for l in fin.readlines():
        words.append(l.strip())
    fin.close()
    word_map = {words[i]: i for i in range(len(words))}
    pad_code = word_map['<eos>']

    content_code = []
    fin = open(text_path, 'r', encoding='UTF-8')
    for l in fin.readlines():
        info = l.strip().split(' ')
        doc_code = [word_map[w] for w in info]
        if len(doc_code) > max_len:
            doc_code = doc_code[:max_len]
        else:
            doc_code.extend([pad_code for _ in range(max_len - len(doc_code))])
        content_code.append(doc_code)

    return content_code, pad_code


def exclusive_combine(*in_list):
    res = set()
    in_list = list(*in_list)
    for n_l in in_list:
        for i in n_l:
            res.add(i)
    return list(res)


def identity_map(n_list):
    id_dict = {}
    for i in range(len(n_list)):
        id_dict[n_list[i]] = i
    return id_dict


def padding(one_hot_features, max_len, pad_code):
    code = []
    for one_hot_doc in one_hot_features:
        doc_code = torch.where(one_hot_doc)[0].numpy().tolist()
        if len(doc_code) > max_len:
            doc_code = doc_code[0: max_len]
        else:
            doc_code.extend([pad_code for _ in range(max_len - len(doc_code))])
        code.append(doc_code)

    return torch.LongTensor(code).cuda()


def get_batches(node_num, batch_size):
    # np.random.seed(1)
    node_idx = list(range(node_num))
    np.random.shuffle(node_idx)
    num_batches = node_num // batch_size
    batch_list = []

    for n in range(num_batches):
        batch_list.append(node_idx[n * batch_size: (n + 1) * batch_size])

    if node_num > num_batches * batch_size:
        batch_list.append(node_idx[num_batches * batch_size:])

    return batch_list


def adj_list(edges):
    neighbor_dict = {}
    for i,j in edges:
        i, j = i.item(), j.item()
        if i in neighbor_dict:
            neighbor_dict[i].add(j)
        else:
            neighbor_dict[i] = {j}

        if j in neighbor_dict:
            neighbor_dict[j].add(i)
        else:
            neighbor_dict[j] = {i}
    for key in neighbor_dict.keys():
        neighbor_dict[key] = list(neighbor_dict[key])

    return neighbor_dict

