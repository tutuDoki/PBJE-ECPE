import sys
import os

sys.path.append('..')
from os.path import join
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from config import *
from utils.utils import *
import dgl
from collections import defaultdict

torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True


def build_train_data(configs, fold_id, shuffle=True):
    train_dataset = MyDataset(configs, fold_id, data_type='train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=shuffle, collate_fn=bert_batch_preprocessing)
    return train_loader


def build_inference_data(configs, fold_id, data_type):
    dataset = MyDataset(configs, fold_id, data_type)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=configs.batch_size,
                                              shuffle=False, collate_fn=bert_batch_preprocessing)
    return data_loader


class MyDataset(Dataset):
    def __init__(self, configs, fold_id, data_type, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.split = configs.split
        self.max_token_len = configs.max_token_len
        self.data_type = data_type
        self.train_file = join('..', data_dir, self.split, TRAIN_FILE % fold_id)
        self.valid_file = join('..', data_dir, self.split, VALID_FILE % fold_id)
        self.test_file = join('..', data_dir, self.split, TEST_FILE % fold_id)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs
        self.K = configs.K
        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.category2idx = {'happiness': 1, 'sadness': 2, 'anger': 3, 'disgust': 4, 'surprise': 5, 'fear': 6,
                             'other': 7}
        self.doc_couples_list, self.y_emotions_list, self.y_causes_list, \
        self.doc_len_list, self.doc_id_list, \
        self.bert_token_idx_list, self.bert_clause_idx_list, self.bert_clause_sep_idx_list, self.bert_segments_idx_list, \
        self.bert_token_lens_list, self.graph_list, self.pair_graph_list, self.emo_pos_list, self.cau_pos_list, \
        self.y_emotion_category_list = self.read_data_file(self.data_type)

    def __len__(self):
        return len(self.y_emotions_list)

    def __getitem__(self, idx):
        doc_couples, y_emotions, y_causes = self.doc_couples_list[idx], self.y_emotions_list[idx], self.y_causes_list[idx]
        doc_len, doc_id = self.doc_len_list[idx], self.doc_id_list[idx]
        graph, pair_graph = self.graph_list[idx], self.pair_graph_list[idx]
        emo_pos, cau_pos = self.emo_pos_list[idx], self.cau_pos_list[idx]
        bert_token_idx, bert_clause_idx, bert_clause_sep_idx = self.bert_token_idx_list[idx], self.bert_clause_idx_list[idx], \
                                                               self.bert_clause_sep_idx_list[idx]
        bert_segments_idx, bert_token_lens = self.bert_segments_idx_list[idx], self.bert_token_lens_list[idx]
        y_emotion_category = self.y_emotion_category_list[idx]
        if bert_token_lens > self.max_token_len:
            bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens, \
            doc_couples, y_emotions, y_causes, doc_len = self.token_trunk(bert_token_idx, bert_clause_idx,
                                                                          bert_clause_sep_idx, bert_segments_idx,
                                                                          bert_token_lens, doc_couples, y_emotions,
                                                                          y_causes, doc_len)

        bert_token_idx = torch.LongTensor(bert_token_idx)
        bert_segments_idx = torch.LongTensor(bert_segments_idx)
        bert_clause_idx = torch.LongTensor(bert_clause_idx)
        bert_clause_sep_idx = torch.LongTensor(bert_clause_sep_idx)

        assert doc_len == len(y_emotions)
        return doc_couples, y_emotions, y_causes, doc_len, doc_id, bert_token_idx, bert_segments_idx, bert_clause_idx, \
               bert_clause_sep_idx, bert_token_lens, graph, pair_graph, emo_pos, cau_pos, y_emotion_category

    def get_category_idx(self, category):
        if category in self.category2idx:
            return self.category2idx[category]
        return self.category2idx['other']

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        doc_id_list = []
        doc_len_list = []
        doc_couples_list = []
        y_emotions_list, y_causes_list = [], []
        y_emotion_category_list = []
        bert_token_idx_list = []
        bert_clause_idx_list = []
        bert_clause_sep_idx_list = []
        bert_segments_idx_list = []
        bert_token_lens_list = []
        graph_list = []
        pair_graph_list = []
        emo_pos_list = []
        cau_pos_list = []
        memory = {}
        emotional_clauses = read_b(os.path.join('..', DATA_DIR, SENTIMENTAL_CLAUSE_DICT))
        data_list = read_json(data_file)
        # 读取json文档中的各种信息
        for doc in data_list:
            doc_id = doc['doc_id']
            doc_len = doc['doc_len']
            doc_couples = doc['pairs']
            doc_emotions, doc_causes = zip(*doc_couples)
            doc_id_list.append(doc_id)
            doc_len_list.append(doc_len)
            doc_couples = list(map(lambda x: list(x), doc_couples))
            doc_couples_list.append(doc_couples)
            y_emotions, y_causes = [], []
            doc_clauses = doc['clauses']
            y_emotions_category = [0] * doc_len
            # 对每个pair的两个句子获取对应的情感index
            for couple in doc_couples:
                emo, cau = couple[0], couple[1]
                emo_category_idx = self.get_category_idx(doc_clauses[emo - 1]['emotion_category'].strip().lower())
                y_emotions_category[emo - 1] = emo_category_idx
                y_emotions_category[cau - 1] = emo_category_idx
            # 构建图
            graph, pair_graph, emo_pos, cau_pos = construct_graph(doc_len, self.K, memory)
            assert max(emo_pos) + 1 == doc_len
            graph_list.append(graph)
            pair_graph_list.append(pair_graph)
            emo_pos_list.append(emo_pos)
            cau_pos_list.append(cau_pos)

            bert_doc_token = []
            for i, clause in enumerate(doc_clauses):
                bert_doc_token.append(
                    self.bert_tokenizer.encode(clause['clause'].strip(), add_special_tokens=False))
            doc_bert_token_len = sum([len(x) for x in bert_doc_token])
            max_clause_len = self.max_token_len // doc_len - 2
            indexed_tokens = []
            cut_flag = doc_bert_token_len + 2 * doc_len > self.max_token_len
            for i in range(doc_len):
                emotion_label = int(i + 1 in doc_emotions)
                cause_label = int(i + 1 in doc_causes)
                y_emotions.append(emotion_label)
                y_causes.append(cause_label)

                clause = doc_clauses[i]
                clause_id = clause['clause_id']
                assert int(clause_id) == i + 1
                if cut_flag and len(bert_doc_token[i]) > max_clause_len:
                    bert_doc_token[i] = bert_doc_token[i][:max_clause_len]
                indexed_tokens.extend([101] + bert_doc_token[i] + [102])

            clause_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
            clause_sep_indices = [i for i, x in enumerate(indexed_tokens) if x == 102]
            doc_token_len = len(indexed_tokens)
            assert doc_token_len <= self.max_token_len

            segments_ids = []
            segments_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
            segments_indices.append(len(indexed_tokens))
            for i in range(len(segments_indices) - 1):
                semgent_len = segments_indices[i + 1] - segments_indices[i]
                if i % 2 == 0:
                    segments_ids.extend([0] * semgent_len)
                else:
                    segments_ids.extend([1] * semgent_len)

            assert len(clause_indices) == doc_len
            assert len(segments_ids) == len(indexed_tokens)
            bert_token_idx_list.append(indexed_tokens)
            bert_clause_idx_list.append(clause_indices)
            bert_clause_sep_idx_list.append(clause_sep_indices)
            bert_segments_idx_list.append(segments_ids)
            bert_token_lens_list.append(doc_token_len)

            y_emotions_list.append(y_emotions)
            y_causes_list.append(y_causes)
            y_emotion_category_list.append(y_emotions_category)

        return doc_couples_list, y_emotions_list, y_causes_list, doc_len_list, doc_id_list, bert_token_idx_list, \
               bert_clause_idx_list, bert_clause_sep_idx_list, bert_segments_idx_list, bert_token_lens_list, graph_list,\
               pair_graph_list, emo_pos_list, cau_pos_list, y_emotion_category_list

    def token_trunk(self, bert_token_idx, bert_clause_idx, bert_clause_sep_idx, bert_segments_idx, bert_token_lens,
                    doc_couples, y_emotions, y_causes, doc_len):
        # TODO: cannot handle some extreme cases now
        emotion, cause = doc_couples[0]
        if emotion > doc_len / 2 and cause > doc_len / 2:
            i = 0
            while True:
                temp_bert_token_idx = bert_token_idx[bert_clause_idx[i]:]
                if len(temp_bert_token_idx) <= self.max_token_len:
                    cls_idx = bert_clause_idx[i]
                    bert_token_idx = bert_token_idx[cls_idx:]
                    bert_segments_idx = bert_segments_idx[cls_idx:]
                    bert_clause_idx = [p - cls_idx for p in bert_clause_idx[i:]]
                    doc_couples = [[emotion - i, cause - i]]
                    y_emotions = y_emotions[i:]
                    y_causes = y_causes[i:]
                    doc_len = doc_len - i
                    break
                i = i + 1
        if emotion < doc_len / 2 and cause < doc_len / 2:
            i = doc_len - 1
            while True:
                temp_bert_token_idx = bert_token_idx[:bert_clause_idx[i]]
                if len(temp_bert_token_idx) <= self.max_token_len:
                    cls_idx = bert_clause_idx[i]
                    bert_token_idx = bert_token_idx[:cls_idx]
                    bert_segments_idx = bert_segments_idx[:cls_idx]
                    bert_clause_idx = bert_clause_idx[:i]
                    y_emotions = y_emotions[:i]
                    y_causes = y_causes[:i]
                    doc_len = i
                    break
                i = i - 1
        return bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens, \
               doc_couples, y_emotions, y_causes, doc_len


def bert_batch_preprocessing(batch):
    doc_couples_b, y_emotions_b, y_causes_b, doc_len_b, doc_id_b, bert_token_b, bert_segment_b, bert_clause_b, \
    bert_clause_sep_b, bert_token_lens_b, graphs, pair_graphs, emo_pos, cau_pos, y_emotion_category_b = zip(*batch)

    y_mask_b, y_emotions_b, y_causes_b = pad_docs(doc_len_b, y_emotions_b, y_causes_b)
    adj_b = pad_matrices(doc_len_b)
    bert_token_b = pad_sequence(bert_token_b, batch_first=True, padding_value=0)
    bert_segment_b = pad_sequence(bert_segment_b, batch_first=True, padding_value=0)
    bert_clause_b = pad_sequence(bert_clause_b, batch_first=True, padding_value=0)
    bert_clause_sep_b = pad_sequence(bert_clause_sep_b, batch_first=True, padding_value=0)

    bsz, max_len = bert_token_b.size()
    bert_masks_b = np.zeros([bsz, max_len], dtype=np.float)
    for index, seq_len in enumerate(bert_token_lens_b):
        bert_masks_b[index][:seq_len] = 1

    bert_masks_b = torch.FloatTensor(bert_masks_b)
    assert bert_segment_b.shape == bert_token_b.shape
    assert bert_segment_b.shape == bert_masks_b.shape

    return np.array(doc_len_b), np.array(adj_b), np.array(y_emotions_b), np.array(y_causes_b), np.array(y_mask_b), \
           doc_couples_b, doc_id_b, bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, bert_clause_sep_b, \
           list(graphs), list(pair_graphs), list(emo_pos), list(cau_pos), list(y_emotion_category_b)


def pad_docs(doc_len_b, y_emotions_b, y_causes_b):
    max_doc_len = max(doc_len_b)

    y_mask_b, y_emotions_b_, y_causes_b_ = [], [], []
    for y_emotions, y_causes in zip(y_emotions_b, y_causes_b):
        y_emotions_ = pad_list(y_emotions, max_doc_len, -1)
        y_causes_ = pad_list(y_causes, max_doc_len, -1)
        y_mask = list(map(lambda x: 0 if x == -1 else 1, y_emotions_))

        y_mask_b.append(y_mask)
        y_emotions_b_.append(y_emotions_)
        y_causes_b_.append(y_causes_)

    return y_mask_b, y_emotions_b_, y_causes_b_


def pad_matrices(doc_len_b):
    N = max(doc_len_b)
    adj_b = []
    for doc_len in doc_len_b:
        adj = np.ones((doc_len, doc_len))
        adj = sp.coo_matrix(adj)
        adj = sp.coo_matrix((adj.data, (adj.row, adj.col)), shape=(N, N), dtype=np.float32)
        adj_b.append(adj.toarray())
    return adj_b

# 对每个batch不等长的句子数量进行padding
def pad_list(element_list, max_len, pad_mark):
    element_list_pad = element_list[:]
    pad_mark_list = [pad_mark] * (max_len - len(element_list))
    element_list_pad.extend(pad_mark_list)
    return element_list_pad


def construct_graph(seq_len, k, memory):
    if seq_len in memory:
        return memory[seq_len]
    d = defaultdict(list)
    for i in range(seq_len):
        d[('sentence_emo', 'sl_emo', 'sentence_emo')].append((i, i))
        d[('sentence_cau', 'sl_cau', 'sentence_cau')].append((i, i))
        for j in range(i + 1, seq_len):
            d[('sentence_emo', 'ss_emo', 'sentence_emo')].append((i, j))
            d[('sentence_emo', 'ss_emo', 'sentence_emo')].append((j, i))
            d[('sentence_cau', 'ss_cau', 'sentence_cau')].append((i, j))
            d[('sentence_cau', 'ss_cau', 'sentence_cau')].append((j, i))
    base_idx = np.arange(0, seq_len)
    emo_pos = np.concatenate([base_idx.reshape(-1, 1)] * seq_len, axis=1).reshape(1, -1)[0]
    cau_pos = np.concatenate([base_idx] * seq_len, axis=0)
    rel_pos = cau_pos - emo_pos
    rel_pos = torch.LongTensor(rel_pos).to(DEVICE)
    emo_pos = torch.LongTensor(emo_pos).to(DEVICE)
    cau_pos = torch.LongTensor(cau_pos).to(DEVICE)
    if seq_len > k + 1:
        rel_mask = np.array(list(map(lambda x: -k <= x <= k, rel_pos.tolist())), dtype=np.int)
        rel_mask = torch.BoolTensor(rel_mask).to(DEVICE)
        rel_pos = rel_pos.masked_select(rel_mask)
        emo_pos = emo_pos.masked_select(rel_mask)
        cau_pos = cau_pos.masked_select(rel_mask)
    count = 0
    pair_graph = dgl.heterograph(construct_pair_graph(seq_len, emo_pos, cau_pos))

    for emo, cau in zip(emo_pos.tolist(), cau_pos.tolist()):
        d[('sentence_emo', 'spe', 'pair')].append((emo, count))
        d[('sentence_cau', 'spc', 'pair')].append((cau, count))
        d[('pair', 'spe', 'sentence_emo')].append((count, emo))
        d[('pair', 'spc', 'sentence_cau')].append((count, cau))
        d[('document', 'doc', 'pair')].append((0, count))
        d[('pair', 'doc', 'document')].append((count, 0))
        d[('pair', 'pl', 'pair')].append((count, count))
        count += 1
    for i in range(seq_len):
        d[('sentence_emo', 'doc', 'document')].append((i, 0))
        d[('document', 'doc', 'sentence_emo')].append((0, i))
        d[('sentence_cau', 'doc', 'document')].append((i, 0))
        d[('document', 'doc', 'sentence_cau')].append((0, i))

    d[('document', 'dl', 'document')].append((0, 0))

    graph = dgl.heterograph(d)
    memory[seq_len] = graph, pair_graph, emo_pos, cau_pos
    return graph, pair_graph, emo_pos, cau_pos


def construct_pair_graph(seq_len, emo_pos, cau_pos):
    d = defaultdict(list)
    d[('pair', 'ppe', 'pair')] = []
    d[('pair', 'ppc', 'pair')] = []
    indices = torch.arange(len(emo_pos)).to(DEVICE)
    d[('pair', 'pl', 'pair')] = [(x, x) for x in indices.tolist()]
    for x in range(seq_len):
        emo_mask = emo_pos == x
        cau_mask = cau_pos == x
        emo_indices = indices.masked_select(emo_mask)
        cau_indices = indices.masked_select(cau_mask)
        d[('pair', 'ppe', 'pair')].extend(construct_full_graph(emo_indices))
        d[('pair', 'ppc', 'pair')].extend(construct_full_graph(cau_indices))
    return d


def construct_full_graph(idx):
    res = []
    idx = idx.cpu().numpy()
    length = len(idx)
    left = np.concatenate([idx.reshape(-1, 1)] * length, axis=1).reshape(1, -1)[0].tolist()
    right = np.concatenate([idx] * length, axis=0).tolist()
    for l, r in zip(left, right):
        if l != r:
            res.append((l, r))
    return res
