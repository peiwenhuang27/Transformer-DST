from utils.data_utils import prepare_dataset,CrossWozDataset, wrap_into_tensor
from utils.data_utils import make_slot_meta, domain2id, OP_SET, make_turn_label, postprocessing
from utils.eval_utils import compute_prf, compute_acc, per_domain_join_accuracy
from pytorch_transformers import BertTokenizer, BertConfig

from model import TransformerDST
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import random
import numpy as np
import os
import sys
import time
import argparse
import json
from copy import deepcopy

from utils.slot_meta import getSlotMeta


def main(args):
    if args.cuda_idx is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_idx
        print('CUDA VISIBLE DEVICES: ', os.environ['CUDA_VISIBLE_DEVICES'])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    print('CUDA DEVICE COUNT: ', n_gpu)

    # ontology = json.load(open(os.path.join(args.data_root, args.ontology_data)))
    # slot_meta, _ = make_slot_meta(ontology)
    slot_meta = getSlotMeta()
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)
    test_path = os.path.join(args.data_root, "test.pt")
    if args.op_code in ['4', '5']:
        test_path = os.path.join(args.data_root, "test_request.pt")
    if not os.path.exists(test_path):
        data = prepare_dataset(os.path.join(args.data_root, args.test_data), tokenizer, slot_meta, 
                               args.n_history, args.max_seq_length, args.op_code)
        torch.save(data, test_path)
    else:
        data = torch.load(test_path)

    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = 0.1
    op2id = OP_SET[args.op_code]
    # model = TransformerDST(model_config, len(op2id), len(domain2id), op2id['update'])
    dec_config = args
    type_vocab_size = 4
    model = TransformerDST(model_config, dec_config, len(op2id), len(domain2id),
                           op2id['update'],
                           tokenizer.convert_tokens_to_ids(['[MASK]'])[0],
                           tokenizer.convert_tokens_to_ids(['[SEP]'])[0],
                           tokenizer.convert_tokens_to_ids(['[PAD]'])[0],
                           tokenizer.convert_tokens_to_ids(['-'])[0],
                           type_vocab_size, False)
    ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    model.to(device)

    if args.eval_all:
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         False, False, False)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         False, False, True)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         False, True, False)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         False, True, True)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         True, False, False)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         True, True, False)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         True, False, True)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         True, True, True)
    else:
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         args.gt_op, args.gt_p_state, args.gt_gen)


def model_evaluation(model, test_data, tokenizer, slot_meta, epoch, op_code='4',
                     is_gt_op=False, is_gt_p_state=False, is_gt_gen=False, use_full_slot=False, use_dt_only=False, no_dial=False, use_cls_only=False, n_gpu=1):

    device = torch.device('cuda' if n_gpu else 'cpu')

    model.eval()
    op2id = OP_SET[op_code]
    id2op = {v: k for k, v in op2id.items()}
    id2domain = {v: k for k, v in domain2id.items()}

    slot_turn_acc, joint_acc, slot_F1_pred, slot_F1_count = 0, 0, 0, 0
    final_joint_acc, final_count, final_slot_F1_pred, final_slot_F1_count = 0, 0, 0, 0
    op_acc, op_F1, op_F1_count = 0, {k: 0 for k in op2id}, {k: 0 for k in op2id}
    all_op_F1_count = {k: 0 for k in op2id}

    tp_dic = {k: 0 for k in op2id}
    fn_dic = {k: 0 for k in op2id}
    fp_dic = {k: 0 for k in op2id}

    results = {}
    last_dialog_state = {}
    wall_times = []

    # TEST
    gold_ops_tmp = []

    start_time = time.time()
    for di, i in enumerate(test_data):
        if (di+1) % 1000 == 0:
            print("{:}, {:.1f}min".format(di, (time.time()-start_time)/60))
            sys.stdout.flush()

        if i.turn_id == 0:
            last_dialog_state = {}

        if is_gt_p_state is False:
            i.last_dialog_state = deepcopy(last_dialog_state)
            i.make_instance(tokenizer, word_dropout=0.)
        else:  # ground-truth previous dialogue state as last_dialog_state
            last_dialog_state = deepcopy(i.gold_p_state)
            i.last_dialog_state = deepcopy(last_dialog_state)
            i.make_instance(tokenizer, word_dropout=0.)

        id2ds = {}
        for id, s in enumerate(i.slot_meta):
            k = s.split('-')
            # print(k)  # e.g. ['attraction', 'area']
            id2ds[id] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join(k + ['-'])))

        # do not use input_ids_g, ... -> why?
        tensor_list = wrap_into_tensor([i], pad_id=tokenizer.convert_tokens_to_ids(['[PAD]'])[0],
                                       slot_id=tokenizer.convert_tokens_to_ids(['[SLOT]'])[0])[:4]
        tensor_list = [t.to(device) for t in tensor_list]
        input_ids_p, segment_ids_p, input_mask_p, state_position_ids = tensor_list

        d_gold_op, _, _ = make_turn_label(slot_meta, last_dialog_state, i.gold_state, tokenizer, op_code, dynamic=True)
        gold_op_ids = torch.LongTensor([d_gold_op]).to(device)

        start = time.perf_counter()

        MAX_LENGTH = 9
        if n_gpu > 1:
            model.module.decoder.min_len = 1  # just ask the decoder to generate at least a token (notice that [SEP] is included)
        else:
            model.decoder.min_len = 1

        with torch.no_grad():
            # ground-truth state operation
            gold_op_inputs = gold_op_ids if is_gt_op else None

            if n_gpu > 1:
                d, s, generated = model.module.output(input_ids_p, segment_ids_p, input_mask_p,
                                               state_position_ids, i.diag_len, op_ids=gold_op_inputs,
                                               gen_max_len=MAX_LENGTH, use_full_slot=use_full_slot, use_dt_only=use_dt_only, diag_1_len=i.diag_1_len,
                                                      no_dial=no_dial, use_cls_only=use_cls_only, i_dslen_map=i.i_dslen_map)
            else:
                # op_ids if not None: ground truth op labels, used in decoder to check slots to update
                d, s, generated = model.output(input_ids_p, segment_ids_p, input_mask_p,
                                               state_position_ids, i.diag_len, op_ids=gold_op_inputs, gen_max_len=MAX_LENGTH,
                                               use_full_slot=use_full_slot, use_dt_only=use_dt_only, diag_1_len=i.diag_1_len,
                                               no_dial=no_dial, use_cls_only=use_cls_only, i_dslen_map=i.i_dslen_map)

        _, op_ids = s.view(-1, len(op2id)).max(-1)

        if is_gt_op:
            pred_ops = [id2op[a] for a in gold_op_ids[0].tolist()]
        else:
            pred_ops = [id2op[a] for a in op_ids.tolist()]
        gold_ops = [id2op[a] for a in d_gold_op]

        # TEST
        gold_ops_tmp += gold_ops
        #

        if is_gt_gen:
            # ground_truth generation
            gold_gen = {'-'.join(ii.split('-')[:2]): ii.split('-')[-1] for ii in i.gold_state}
        else:
            gold_gen = {}

        # print("generated:", generated)
        # print("last_dialog_state:", last_dialog_state)


        generated, last_dialog_state = postprocessing(slot_meta, pred_ops, last_dialog_state,
                                                      generated, tokenizer, op_code, gold_gen)



        end = time.perf_counter()
        wall_times.append(end - start)
        pred_state = []
        for k, v in last_dialog_state.items():
            pred_state.append('-'.join([k, v]))

        if set(pred_state) == set(i.gold_state):
            joint_acc += 1
        key = str(i.id) + '_' + str(i.turn_id)
        results[key] = [pred_state, i.gold_state]

        # Compute prediction slot accuracy
        temp_acc = compute_acc(set(i.gold_state), set(pred_state), slot_meta)
        slot_turn_acc += temp_acc

        # Compute prediction F1 score
        temp_f1, temp_r, temp_p, count = compute_prf(i.gold_state, pred_state)
        slot_F1_pred += temp_f1
        slot_F1_count += count

        # Compute operation accuracy
        temp_acc = sum([1 if p == g else 0 for p, g in zip(pred_ops, gold_ops)]) / len(pred_ops)
        op_acc += temp_acc

        if i.is_last_turn:
            final_count += 1
            if set(pred_state) == set(i.gold_state):
                final_joint_acc += 1

            final_slot_F1_pred += temp_f1
            final_slot_F1_count += count

        # Compute operation F1 score
        for p, g in zip(pred_ops, gold_ops):
            all_op_F1_count[g] += 1
            if p == g:
                tp_dic[g] += 1
                op_F1_count[g] += 1
            else:
                fn_dic[g] += 1
                fp_dic[p] += 1

    # TEST
    from collections import Counter
    gold_ops_cnt = Counter(gold_ops_tmp)
    print("gold_ops_cnt: ", gold_ops_cnt)
    print("fn_dic: ", fn_dic)
    print("fp_dic: ", fp_dic)
    #
    joint_acc_score = joint_acc / len(test_data)
    turn_acc_score = slot_turn_acc / len(test_data)
    slot_F1_score = slot_F1_pred / slot_F1_count
    op_acc_score = op_acc / len(test_data)
    final_joint_acc_score = final_joint_acc / final_count
    final_slot_F1_score = final_slot_F1_pred / final_slot_F1_count
    latency = np.mean(wall_times) * 1000
    op_F1_score = {}
    for k in op2id.keys():
        tp = tp_dic[k]
        fn = fn_dic[k]
        fp = fp_dic[k]
        precision = tp / (tp+fp) if (tp+fp) != 0 else 0
        recall = tp / (tp+fn) if (tp+fn) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        op_F1_score[k] = F1

    print("------------------------------")
    print('op_code: %s, is_gt_op: %s, is_gt_p_state: %s, is_gt_gen: %s' % \
          (op_code, str(is_gt_op), str(is_gt_p_state), str(is_gt_gen)))
    print("Epoch %d joint accuracy : " % epoch, joint_acc_score)
    print("Epoch %d slot turn accuracy : " % epoch, turn_acc_score)
    print("Epoch %d slot turn F1: " % epoch, slot_F1_score)
    print("Epoch %d op accuracy : " % epoch, op_acc_score)
    print("Epoch %d op F1 : " % epoch, op_F1_score)
    print("Epoch %d op hit count : " % epoch, op_F1_count)
    print("Epoch %d op all count : " % epoch, all_op_F1_count)
    print("Final Joint Accuracy : ", final_joint_acc_score)
    print("Final slot turn F1 : ", final_slot_F1_score)
    print("Latency Per Prediction : %f ms" % latency)
    print("-----------------------------\n")
    json.dump(results, open('preds_%d.json' % epoch, 'w'))

    per_domain_join_accuracy(results, slot_meta)

    scores = {'epoch': epoch, 'joint_acc': joint_acc_score,
              'slot_acc': turn_acc_score, 'slot_f1': slot_F1_score,
              'op_acc': op_acc_score, 'op_f1': op_F1_score, 'final_slot_f1': final_slot_F1_score}
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default='', type=str)
    parser.add_argument("--test_data", default='cleanData/test_dials.json', type=str)
    # parser.add_argument("--ontology_data", default='ontology.json', type=str)
    parser.add_argument("--vocab_path", default='assets/vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default='assets/bert_config_base_chinese.json', type=str)
    parser.add_argument("--model_ckpt_path", default='outputs/dt_e15_b16/model.e7.bin', type=str)
    parser.add_argument("--n_history", default=1, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--op_code", default="4", type=str)
    parser.add_argument("--cuda_idx", default=None, type=str)

    parser.add_argument("--gt_op", default=False, action='store_true')
    parser.add_argument("--gt_p_state", default=False, action='store_true')
    parser.add_argument("--gt_gen", default=False, action='store_true')
    parser.add_argument("--eval_all", default=False, action='store_true')

    # generator
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument("--min_len", default=1, type=int)
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")
    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Ignore the word during forbid_duplicate_ngrams")
    parser.add_argument('--ngram_size', type=int, default=2)

    args = parser.parse_args()
    main(args)