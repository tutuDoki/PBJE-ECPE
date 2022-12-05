import sys, os, warnings, time
import argparse

sys.path.append('..')
warnings.filterwarnings("ignore")
import numpy as np
import torch
from config import *
from data_loader import *
from networks.RGCN import *
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.utils import *
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
dgl.random.seed(TORCH_SEED)
np.random.seed(TORCH_SEED)
os.environ['OMP_NUM_THREADS'] = '1'


def main(configs, fold_id, args):
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    dgl.random.seed(TORCH_SEED)
    dgl.seed(TORCH_SEED)

    train_loader = build_train_data(configs, fold_id=fold_id)
    if configs.split == 'split20':
        valid_loader = build_inference_data(configs, fold_id=fold_id, data_type='valid')
    test_loader = build_inference_data(configs, fold_id=fold_id, data_type='test')
    configs.num_fg_tags = len(train_loader.dataset.category2idx)
    configs.rel_name = train_loader.dataset.graph_list[0].etypes
    model = Network(configs).to(DEVICE)
    params = model.parameters()
    params_bert_id = list(map(id, model.bert.parameters()))
    params_rest = filter(lambda p: id(p) not in params_bert_id, model.parameters())
    assert sum([param.nelement() for param in params]) == \
           sum([param.nelement() for param in model.bert.parameters()]) + sum(
        [param.nelement() for param in params_rest])

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    params = [
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': configs.l2_bert, 'eps': configs.adam_epsilon},
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'eps': configs.adam_epsilon},
        {'params': params_rest,
         'weight_decay': configs.l2}
    ]
    optimizer = AdamW(params, lr=configs.lr)

    num_steps_all = len(train_loader) // configs.gradient_accumulation_steps * configs.epochs
    warmup_steps = int(num_steps_all * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_steps_all)

    model.zero_grad()
    max_ec, max_e, max_c = [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]
    metric_ec, metric_e, metric_c = [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]
    early_stop_flag = None

    emotional_clauses = read_b(os.path.join('..', DATA_DIR, SENTIMENTAL_CLAUSE_DICT))

    for epoch in range(1, configs.epochs + 1):
        with tqdm(desc='Training', total=len(train_loader)) as pbar:
            for train_step, batch in enumerate(train_loader, 1):
                model.train()
                pbar.update()
                doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, bert_token_b, \
                bert_segment_b, bert_masks_b, bert_clause_b, bert_clause_sep_b, graphs, pair_graphs, emo_pos, cau_pos, \
                y_emotion_category_b = batch

                couples_pred, emo_cau_pos, pred_e, pred_c = model(bert_token_b, bert_segment_b, bert_masks_b,
                                                                  bert_clause_b, bert_clause_sep_b, doc_len_b, adj_b,
                                                                  y_mask_b, graphs, pair_graphs, emo_pos, cau_pos,
                                                                  doc_id_b, emotional_clauses)
                loss_e, loss_c = model.loss_pre(pred_e, pred_c, y_emotions_b, y_causes_b, y_mask_b)
                loss_couple, _ = model.loss_rank(couples_pred, emo_cau_pos, doc_couples_b)
                loss = loss_couple + loss_e + loss_c
                loss = loss / configs.gradient_accumulation_steps
                loss.backward()
                pbar.set_description('Training Epochs {}, loss{:.4}'.format(epoch, loss))
                if train_step % configs.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
        with torch.no_grad():
            model.eval()

            if configs.split == 'split10':
                test_ec, test_e, test_c, metric_emo, metric_cau = inference_one_epoch(configs, test_loader, model, emotional_clauses)
                if test_ec[2] > metric_ec[2]:
                    metric_ec, metric_e, metric_c = test_ec, test_e, test_c
            if configs.split == 'split20':
                valid_ec, valid_e, valid_c, metric_emo, metric_cau = inference_one_epoch(configs, valid_loader, model, emotional_clauses)
                test_ec, test_e, test_c, metric_emo, metric_cau = inference_one_epoch(configs, test_loader, model, emotional_clauses)
                if valid_ec[2] > max_ec[2]:
                    early_stop_flag = 1
                    max_ec, max_e, max_c = valid_ec, valid_e, valid_c
                    metric_ec, metric_e, metric_c = test_ec, test_e, test_c
                else:
                    early_stop_flag += 1
    return metric_ec, metric_e, metric_c


def inference_one_batch(configs, batch, model, emotional_clauses):
    doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, bert_clause_sep_b, graphs, pair_graphs, emo_pos, cau_pos, y_emotion_category_b = batch

    couples_pred, emo_cau_pos, pred_e, pred_c = model(bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b,
                                                      bert_clause_sep_b, doc_len_b, adj_b, y_mask_b, graphs,
                                                      pair_graphs,emo_pos, cau_pos, doc_id_b, emotional_clauses)
    loss_e, loss_c, pred_e, pred_c, true_e, true_c = model.loss_pre(pred_e, pred_c, y_emotions_b, y_causes_b, y_mask_b,
                                                                    test=True)
    loss_couple, doc_couples_pred_b = model.loss_rank(couples_pred, emo_cau_pos, doc_couples_b, test=True)
    return to_np(loss_couple), to_np(loss_e), to_np(loss_c), \
           doc_couples_b, doc_couples_pred_b, doc_id_b, detection(pred_e, doc_len_b), detection(pred_c, doc_len_b) \
        , true_e.long().tolist(), true_c.long().tolist()


def detection(batch_pred, doc_len):
    if batch_pred is None:
        return []
    batch_pred = list(batch_pred.split(doc_len.tolist()))
    batch_pred_res = []
    for doc_pred, length in zip(batch_pred, doc_len):
        k = (min(3, int(length)))
        top_values, top_indices = doc_pred.topk(k, dim=-1)
        conditions = top_values.sigmoid() > 0.5
        conditions[0] = True
        result_indices = top_indices.masked_select(conditions)
        pred = torch.zeros(doc_pred.shape).long()
        pred[result_indices] = 1
        batch_pred_res.append(pred)
    return torch.cat(batch_pred_res).tolist()


def inference_one_epoch(configs, batches, model, emotional_clauses):
    doc_id_all, doc_couples_all, doc_couples_pred_all, pred_emo_all, pred_cau_all, y_emo_all, y_cau_all = [], [], [], [], [], [], []
    for batch in batches:
        _, _, _, doc_couples, doc_couples_pred, doc_id_b, pred_e, pred_c, y_emotions, y_causes = inference_one_batch(
            configs, batch, model, emotional_clauses)
        doc_id_all.extend(doc_id_b)
        doc_couples_all.extend(doc_couples)
        doc_couples_pred_all.extend(doc_couples_pred)
        pred_emo_all.extend(pred_e)
        pred_cau_all.extend(pred_c)
        y_emo_all.extend(y_emotions)
        y_cau_all.extend(y_causes)

    doc_couples_pred_all = pair_extraction(doc_id_all, doc_couples_pred_all, doc_couples_all)
    metric_ec, metric_e, metric_c = eval_func(doc_couples_all, doc_couples_pred_all)
    metric_emo = (f1_score(y_emo_all, pred_emo_all), precision_score(y_emo_all, pred_emo_all),
                  recall_score(y_emo_all, pred_emo_all))
    metric_cau = (f1_score(y_cau_all, pred_cau_all), precision_score(y_cau_all, pred_cau_all),
                  recall_score(y_cau_all, pred_cau_all))
    return metric_ec, metric_e, metric_c, metric_emo, metric_cau


def pair_extraction(doc_ids, couples_pred, doc_couples_all, threshold=0.5):
    couples_pred_filtered = []
    for i, (doc_id, couples_pred_i, doc_couple) in enumerate(zip(doc_ids, couples_pred, doc_couples_all)):
        top1, top1_prob = couples_pred_i[0][0], couples_pred_i[0][1]
        couples_pred_i_filtered = [top1]
        for couple in couples_pred_i[1:]:
            if logistic(couple[1]) > threshold:
                couples_pred_i_filtered.append(couple[0])
        couples_pred_filtered.append(couples_pred_i_filtered)
    return couples_pred_filtered


if __name__ == '__main__':
    configs = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='split10', type=str)
    args = parser.parse_args()
    configs.split = args.split
    if args.split == 'split10':
        n_folds = 10
    elif args.split == 'split20':
        n_folds = 20
    else:
        print('Unknown data split.')
        exit()

    metric_folds = {'ecp': [], 'emo': [], 'cau': []}
    for fold_id in range(1, n_folds + 1):
        print('===== fold {} ====='.format(fold_id))
        sys.stdout.flush()
        metric_ec, metric_e, metric_c = main(configs, fold_id, args)
        print('F_ecp: {}\n'.format(float_n(metric_ec[2])))
        metric_folds['ecp'].append(metric_ec)
        metric_folds['emo'].append(metric_e)
        metric_folds['cau'].append(metric_c)
        sys.stdout.flush()

    metric_ec_final = np.mean(np.array(metric_folds['ecp']), axis=0).tolist()
    metric_e_final = np.mean(np.array(metric_folds['emo']), axis=0).tolist()
    metric_c_final = np.mean(np.array(metric_folds['cau']), axis=0).tolist()

    print('=========== Average ===========')
    print('F_ecp: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_ec_final[2]), float_n(metric_ec_final[0]), float_n(metric_ec_final[1])))
    print('F_emo: {}, P_emo: {}, R_emo: {}'.format(float_n(metric_e_final[2]), float_n(metric_e_final[0]), float_n(metric_e_final[1])))
    print('F_cau: {}, P_cau: {}, R_cau: {}'.format(float_n(metric_c_final[2]), float_n(metric_c_final[0]), float_n(metric_c_final[1])))
    print('acc_num: {}, pred_num: {}, true_num: {}'.format(metric_ec_final[3], metric_ec_final[4], metric_ec_final[5]))
    print(str(configs))
    write_b({'ecp': metric_ec_final, 'emo': metric_e_final, 'cau': metric_c_final, 'configs': str(configs)},
            '{}_{}_{}_metrics.pkl'.format(TORCH_SEED, metric_ec_final[2], configs.split))
