import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE
from transformers import BertModel
from torch.nn.init import uniform_


class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.configs = configs
        self.bert = BertModel.from_pretrained(configs.bert_cache_path, output_attentions=True)
        rel_name = ['spe', 'spc', 'ss_emo', 'ss_cau', 'doc', 'sl_emo', 'sl_cau', 'pl', 'dl']
        self.lstm = DynamicLSTM(configs.feat_dim, int(configs.feat_dim / 2))
        self.doc_extend = nn.Linear(configs.feat_dim, configs.feat_dim * 2)
        self.doc_project = nn.Linear(configs.feat_dim * 2, configs.feat_dim)
        self.activation = nn.ReLU()
        self.pair_project = nn.Linear(configs.feat_dim * 2, configs.feat_dim)
        self.pair_generate = nn.Embedding(2 * configs.K + 1, configs.feat_dim)
        self.emo_embedding_model = EmotionEmbedding(configs)
        self.pos_embedding_model = PositionEmbedding(configs)
        self.GCN_layers = nn.ModuleList([RelGraphConvLayer(configs.feat_dim, configs.feat_dim, rel_name,
                                                           num_bases=len(rel_name), activation=self.activation,
                                                           self_loop=True, dropout=configs.dropout)
                                         for i in range(configs.layers)])
        self.dropout = nn.Dropout(p=0.2)
        self.emo_attention = nn.Linear(configs.feat_dim * 2, 1)
        self.cau_attention = nn.Linear(configs.feat_dim * 2, 1)

        self.pred_e_layer = nn.Linear(configs.feat_dim, 1)
        self.pred_c_layer = nn.Linear(configs.feat_dim, 1)
        self.pred_pair_layer1 = nn.Linear(configs.feat_dim * 3, configs.feat_dim * 3)
        self.pred_pair_layer2 = nn.Linear(configs.feat_dim * 3, 1)

        self.document_valve = nn.Linear(configs.feat_dim, 1)

    def forward(self, bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, bert_clause_sep_b, doc_len, adj,
                y_mask_b, graphs, pair_graphs, emo_pos, cau_pos, doc_id_b, emotional_clauses):
        batch_size = len(doc_len)
        bert_output = self.bert(input_ids=bert_token_b.to(DEVICE),
                                attention_mask=bert_masks_b.to(DEVICE),
                                token_type_ids=bert_segment_b.to(DEVICE))

        word_embedding_by_sentence = self.get_word_embedding_bert(bert_output, bert_clause_b, bert_clause_sep_b, batch_size, doc_len)
        avg_sen_embedding = torch.stack([torch.mean(word_embedding_by_sentence[i], dim=0) for i in range(len(word_embedding_by_sentence))])

        sentence_temp = avg_sen_embedding.split(doc_len.tolist())
        document = torch.stack([torch.mean(sentence_temp[i], dim=0) for i in range(batch_size)])

        sentence = avg_sen_embedding
        sentence_emo = sentence
        sentence_cau = sentence.clone()

        tmp_sentence = sentence.split(doc_len.tolist())
        pair = [torch.cat([tmp_sentence[i].index_select(0, emo_pos[i]), tmp_sentence[i].index_select(0, cau_pos[i])], -1)
                for i in range(batch_size)]

        pair = torch.cat(pair)
        pair = self.pair_project(pair)

        position_tag = []
        for i in range(batch_size):
            for j in range(len(emo_pos[i])):
                position_tag.append(int(self.configs.K + emo_pos[i][j] - cau_pos[i][j]))
        position_tag = torch.LongTensor(position_tag).to(DEVICE)
        pair = self.pos_embedding_model(pair, position_tag)

        bg = dgl.batch_hetero(graphs).to(DEVICE)
        sentence_emo_list = []
        sentence_cau_list = []
        pair_list = []
        features = {"sentence_emo": sentence_emo, "sentence_cau": sentence_cau, "pair": pair, "document": document}
        for GCN_layer in self.GCN_layers:
            features = GCN_layer(bg, features)
            sentence_emo_list.append(features['sentence_emo'])
            sentence_cau_list.append(features['sentence_cau'])
            pair_list.append(features['pair'])

        sentence_emo = features['sentence_emo']
        sentence_cau = features['sentence_cau']
        pair = features['pair']

        emo_cau_list = []
        emo_cau_cnt = 0
        for i in range(batch_size):
            sen_emo = sentence_emo[emo_cau_cnt: emo_cau_cnt + doc_len[i]]
            sen_cau = sentence_cau[emo_cau_cnt: emo_cau_cnt + doc_len[i]]
            emo_cau_list.append(torch.cat([sen_emo.index_select(0, emo_pos[i]), sen_cau.index_select(0, cau_pos[i])], dim=1))
            emo_cau_cnt += doc_len[i]
        emo_cau = torch.cat(emo_cau_list)
        pair = torch.cat([pair, emo_cau], dim=1)

        pred_e, pred_c = self.pred_e_layer(sentence_emo), self.pred_c_layer(sentence_cau)
        couples_pred = self.pred_pair_layer2(self.activation(self.dropout(self.pred_pair_layer1(pair))))

        emo_cau_pos = []
        for i in range(batch_size):
            emo_cau_pos_i = []
            for emo, cau in zip(emo_pos[i], cau_pos[i]):
                emo_cau_pos_i.append([int(emo + 1), int(cau + 1)])
            emo_cau_pos.append(emo_cau_pos_i)
        return couples_pred.squeeze(-1), emo_cau_pos, pred_e.squeeze(-1), pred_c.squeeze(-1)

    def batched_index_select(self, bert_output, bert_clause_b):
        hidden_state = bert_output[0]
        dummy = bert_clause_b.unsqueeze(2).expand(bert_clause_b.size(0), bert_clause_b.size(1), hidden_state.size(2))
        doc_sents_h = hidden_state.gather(1, dummy)
        attentions_score = bert_output[2][11]
        atten_list = [attentions_score[i, :, bert_clause_b[i], :] for i in range(len(bert_clause_b))]
        for i in range(len(bert_clause_b)):
            atten_list[i] = atten_list[i][:, :, bert_clause_b[i]].permute(1, 2, 0)
        return doc_sents_h, atten_list

    def attention_batch_index_select(self, attentions_output, bert_clause_b):
        pass

    def get_word_embedding_bert(self, bert_output, bert_clause_b, bert_clause_sep_b, batch_size, doc_len):
        assert bert_clause_b.size() == bert_clause_sep_b.size()
        hidden_state = bert_output[0]
        word_embedding_by_sentence = []
        for i in range(batch_size):
            for j in range(doc_len[i]):
                cls = bert_clause_b[i][j]
                sep = bert_clause_sep_b[i][j]
                word_embedding_by_sentence.append(hidden_state[i][cls + 1: sep])

        return word_embedding_by_sentence


    def k_max_pooling(self, h, logits, k, mask):
        batch_size = h.shape[1]
        candidate = logits.masked_fill(mask, -1000).topk(k, -1)[1]
        h_pooled = []
        for i in range(batch_size):
            h_pooled.append(h[candidate[i], i])
        return torch.stack(h_pooled).permute(1, 0, 2)

    def loss_rank(self, couples_pred, emo_cau_pos, doc_couples, test=False):
        couples_true = []
        batch_size = len(emo_cau_pos)
        for i in range(batch_size):
            couples_num = len(emo_cau_pos[i])
            true_indices = [emo_cau_pos[i].index(x) for x in doc_couples[i] if abs(x[0] - x[1]) <= self.configs.K]
            temp = torch.zeros(couples_num)
            temp[true_indices] = 1.
            couples_true.append(temp)
        couples_true = torch.cat(couples_true)
        couples_true = torch.FloatTensor(couples_true).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        loss_couple = criterion(couples_pred, couples_true)
        doc_couples_pred = []
        if test:
            couples_pred = couples_pred.split([len(x) for x in emo_cau_pos])
            for i in range(batch_size):
                couples_pred_i = couples_pred[i]
                if torch.sum(torch.isnan(couples_pred_i)) > 0:
                    k_idx = [0] * 3
                else:
                    _, k_idx = torch.topk(couples_pred_i, k=3, dim=0)
                # (位置，网络输出的得分)
                doc_couples_pred_i = [(emo_cau_pos[i][idx], couples_pred_i[idx].tolist()) for idx in k_idx]
                doc_couples_pred.append(doc_couples_pred_i)
        return loss_couple, doc_couples_pred

    def loss_pre(self, pred_e, pred_c, y_emotions, y_causes, y_mask, test=False):
        y_mask = torch.BoolTensor(y_mask).to(DEVICE)
        y_emotions = torch.FloatTensor(y_emotions).to(DEVICE)
        y_causes = torch.FloatTensor(y_causes).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        true_e = y_emotions.masked_select(y_mask)
        loss_e = criterion(pred_e, true_e)

        true_c = y_causes.masked_select(y_mask)
        loss_c = criterion(pred_c, true_c)
        if test:
            return loss_e, loss_c, pred_e, pred_c, true_e, true_c
        return loss_e, loss_c

    def loss_similarity(self, feature_emo, feature_cau):
        def compute_kernel_bias(vecs):
            """计算kernel和bias
            最后的变换：y = (x + bias).dot(kernel)
            """
            vecs = np.concatenate(vecs, axis=0)
            mu = vecs.mean(axis=0, keepdims=True)
            cov = np.cov(vecs.T)
            u, s, vh = np.linalg.svd(cov)
            W = np.dot(u, np.diag(1/np.sqrt(s)))
            return W, -mu

        def normalize(vecs):
            """标准化
            """
            return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

        def transform_and_normalize(vecs, kernel, bias):
            """应用变换，然后标准化
            """
            if not (kernel is None or bias is None):
                vecs = torch.mm((vecs + bias), kernel)
            return normalize(vecs)

        tmp_emo = torch.split(feature_emo.clone().cpu().detach(), 1, dim=0)
        tmp_cau = torch.split(feature_cau.clone().cpu().detach(), 1, dim=0)
        W, mu = compute_kernel_bias([*tmp_emo, *tmp_cau])
        W = torch.tensor(W, dtype=torch.float32).to(DEVICE)
        mu = torch.tensor(mu, dtype=torch.float32).to(DEVICE)
        feature_cau = transform_and_normalize(feature_cau, W, mu)
        feature_cau = transform_and_normalize(feature_cau, W, mu)

        loss = torch.mean(torch.cosine_similarity(feature_emo, feature_cau))
        return loss

    """分布softmax loss，平均，bug修复，目前sota"""
    def loss_cross(self, pred_e, pred_c, couples_pred, emo_cau_pos, doc_len_b, doc_couples_b):
        pred_e = F.sigmoid(pred_e)
        pred_c = F.sigmoid(pred_c)
        couples_pred = F.sigmoid(couples_pred)
        emo_pair_sum_list = []
        cau_pair_sum_list = []

        pred_e_list = []
        pred_c_list = []
        sen_cnt = 0
        for i in range(len(doc_len_b)):
            pred_e_list.append(F.softmax(pred_e[sen_cnt: sen_cnt + doc_len_b[i]], dim=0))
            pred_c_list.append(F.softmax(pred_c[sen_cnt: sen_cnt + doc_len_b[i]], dim=0))
            sen_cnt += doc_len_b[i]
        pred_e = torch.cat(pred_e_list)
        pred_c = torch.cat(pred_c_list)

        cross_cnt = 0
        for i in range(len(emo_cau_pos)):
            emo_pair_sum = torch.zeros(doc_len_b[i], dtype=torch.float32).to(DEVICE)
            cau_pair_sum = torch.zeros(doc_len_b[i], dtype=torch.float32).to(DEVICE)
            emo_pair_sum_cnt = torch.zeros(doc_len_b[i], dtype=torch.float32).to(DEVICE)
            cau_pair_sum_cnt = torch.zeros(doc_len_b[i], dtype=torch.float32).to(DEVICE)
            for j in range(len(emo_cau_pos[i])):
                emo, cau = emo_cau_pos[i][j]
                emo_pair_sum[emo - 1] += couples_pred[cross_cnt]
                cau_pair_sum[cau - 1] += couples_pred[cross_cnt]
                emo_pair_sum_cnt[emo - 1] += 1
                cau_pair_sum_cnt[emo - 1] += 1
                cross_cnt += 1

            emo_pair_sum = F.softmax(emo_pair_sum / emo_pair_sum_cnt, dim=0)
            cau_pair_sum = F.softmax(cau_pair_sum / cau_pair_sum_cnt, dim=0)
            emo_pair_sum_list.append(emo_pair_sum)
            cau_pair_sum_list.append(cau_pair_sum)

        emo_pair_sum = torch.cat(emo_pair_sum_list, dim=0)
        cau_pair_sum = torch.cat(cau_pair_sum_list, dim=0)
        criterion_emo = nn.L1Loss(reduction='mean')
        criterion_cau = nn.L1Loss(reduction='mean')
        loss_emo_cross = criterion_emo(emo_pair_sum, pred_e)
        loss_cau_cross = criterion_cau(cau_pair_sum, pred_c)
        return loss_emo_cross + loss_cau_cross

class EmotionEmbedding(nn.Module):
    def __init__(self, configs):
        super(EmotionEmbedding, self).__init__()
        self.emotion_embeddings = nn.Embedding(2, configs.feat_dim)
        self.LayerNorm = nn.LayerNorm(configs.feat_dim, eps=configs.epsilon)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x, emotion_tags):
        emotion_embeddings = self.emotion_embeddings(emotion_tags)
        x += emotion_embeddings
        x = self.LayerNorm(x)
        x = self.dropout(x)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, configs):
        super(PositionEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(2 * configs.K + 1, configs.feat_dim)
        self.LayerNorm = nn.LayerNorm(configs.feat_dim, eps=configs.epsilon)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x, position_tags):
        position_embeddings = self.position_embeddings(position_tags)
        x += position_embeddings
        x = self.LayerNorm(x)
        x = self.dropout(x)
        return x


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=True, only_use_last_hidden_state=False, rnn_type='LSTM', attention_pool=False):
        """
        LSTM which can hold variable length sequence, use like TensoFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type
        self.attention_pool = attention_pool
        self.w1 = nn.Parameter(torch.FloatTensor(hidden_size * 2, hidden_size * 2))
        self.b1 = nn.Parameter(torch.FloatTensor(hidden_size * 2))
        self.w2 = nn.Parameter(torch.FloatTensor(hidden_size * 2, 1))
        self._reset_parameters()

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def _reset_parameters(self):
        uniform_(self.w1, -1, 1)
        uniform_(self.w2, -1, 1)
        uniform_(self.b1, -1, 1)

    def sequence_mask(self, length, max_len, mask_value=False):
        """Mask irrelevant entries in sequences."""
        length = torch.tensor(length)
        mask = torch.arange((max_len), dtype=torch.float32)[None, :] < length[:, None]
        return (mask == mask_value).cuda()

    def forward(self, x, x_len, h0=None):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.argsort(-x_len)
        x_unsort_idx = torch.argsort(x_sort_idx).long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx.long()]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)

        # process using the selected RNN
        if self.rnn_type == 'LSTM':
            if h0 is None:
                out_pack, (ht, ct) = self.RNN(x_emb_p, None)
            else:
                out_pack, (ht, ct) = self.RNN(x_emb_p, (h0, h0))
        else:
            if h0 is None:
                out_pack, ht = self.RNN(x_emb_p, None)
            else:
                out_pack, ht = self.RNN(x_emb_p, h0)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]
        ht = torch.transpose(ht, 0, 1)
        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)
            out = out[0]
            out = out[x_unsort_idx]
            """ attention pool"""
            if self.attention_pool:
                max_len = out.shape[1]
                alpha = torch.matmul(torch.tanh(torch.matmul(out, self.w1) + self.b1), self.w2)
                alpha = alpha.reshape(-1, 1, max_len)
                mask = self.sequence_mask(x_len, max_len).reshape(-1, 1, max_len)
                alpha = alpha.masked_fill_(mask, float('-inf')).softmax(-1)
                out = torch.matmul(alpha, out.squeeze(1))
            """unsort: out c"""
            if self.rnn_type == 'LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]
                ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)
