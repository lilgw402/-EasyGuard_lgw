# -*- coding: utf-8 -*-
import random
from itertools import chain

import numpy as np
import torch.distributed as dist
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score


def all_gather_object(outputs):
    # Better for objects on cpu, found OOM Error if on gpu
    total_outputs = [None for i in range(dist.get_world_size())]
    dist.all_gather_object(total_outputs, outputs)
    total_outputs = sum(total_outputs, [])
    return total_outputs


def sample_frames(num_frames, vlen, sample="uniform", fix_start=None):
    """Sample frames.
    num_frames: num of frames to sample
    vlen: total num of video frames
    """
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == "rand":
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == "uniform":
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError
    return frame_idxs


def pr_fix_t(output, labels, thr):
    recall = np.sum(((output >= thr) == labels) * labels) / np.sum(labels == 1)
    precision = np.sum(((output >= thr) == labels) * labels) / np.sum(output >= thr)
    return precision, recall


def p_fix_r(output, labels, fix_r):
    output_sort = output[(-output).argsort()]
    labels_sort = labels[(-output).argsort()]
    num_pos = np.sum(labels == 1)
    recall_sort = np.cumsum(labels_sort) / float(num_pos)
    index = np.abs(recall_sort - fix_r).argmin()
    thr = output_sort[index]
    precision = np.sum(((output >= thr) == labels) * labels) / np.sum(output >= thr)
    return precision, recall_sort[index], thr


def r_fix_p(output, labels, fix_p):
    output_sort = output[(-output).argsort()]
    labels_sort = labels[(-output).argsort()]
    precision_sort = np.cumsum(labels_sort) / np.cumsum(output_sort >= 0)
    index_list = np.where(np.abs(precision_sort - fix_p) < 0.0001)[0]
    # index = np.abs(precision_sort - fix_p).argmin()
    if len(index_list) == 0:
        index_list = np.where(np.abs(precision_sort - fix_p) < 0.001)[0]
        print("decrease thr to 0.001")
    if len(index_list) == 0:
        index_list = np.where(np.abs(precision_sort - fix_p) < 0.01)[0]
        print("decrease thr to 0.01")
    try:
        index = max(index_list)
    except:
        index = np.abs(precision_sort - fix_p).argmin()
    thr = output_sort[index]
    recall = np.sum(((output >= thr) == labels) * labels) / np.sum(labels == 1)
    return precision_sort[index], recall, thr


def print_res(prob, labels):
    print("Precision / Recall / Threshold")
    # precision, recall, thr = p_fix_r(prob, labels, 0.3)
    # print(str(precision * 100)[:5] + " / " + str(recall * 100)[:5] + " / " + str(thr)[:5])
    precision, recall, thr = r_fix_p(prob, labels, 0.7)
    print(str(precision * 100)[:5] + " / " + str(recall * 100)[:5] + " / " + str(thr)[:5])


def compute_accuracy(logits, labels):
    """
    :param logits:  [batch_size,category_num]
    :param labels:  [batch_size]
    :return:
    """
    pred = logits.argmax(axis=1).reshape(-1)
    labels = labels.reshape(-1)
    tp = pred.eq(labels)
    correct = tp.sum().item()
    total = labels.shape[0]
    acc = float(correct) / total
    return acc


def compute_f1_score(logits, labels):
    pred = logits.argmax(axis=1).reshape(-1)
    labels = labels.reshape(-1)
    macro_f1_score = f1_score(labels, pred, average="macro")
    return macro_f1_score


def compute_f1(res, pos_prob=None, ratios=None):
    lv1_names = list(site_label_dict_lv1.keys())
    lv2_names = list(site_label_dict_lv2.keys())

    res["lv1"]["names"] = lv1_names
    res["lv2"]["names"] = lv2_names

    metrics = {}
    for split in res:
        prob_list = res[split]["probs"]
        label_list = res[split]["labels"]
        target_names = res[split]["names"]

        pred_list = [np.argmax(prob) for prob in prob_list]

        pred_list, label_list = list(zip(*[[p, l] for p, l in zip(pred_list, label_list) if l != -1]))
        valid_labels, target_names = list(zip(*[(i, name) for i, name in enumerate(target_names)]))

        # delete_names = ['其他']
        # pred_list, label_list = list(zip(*[[p, l] for p, l in zip(pred_list, label_list) if l != -1]))
        # valid_labels, target_names = list(zip(*[(i, name) for i, name in enumerate(target_names) if name not in delete_names]))

        # rank_zero_info('-'*10+split+'-'*10)
        # rank_zero_info(classification_report(label_list, pred_list, labels=valid_labels, target_names=target_names, digits=4))
        metric = classification_report(
            label_list,
            pred_list,
            labels=valid_labels,
            target_names=target_names,
            digits=4,
            output_dict=True,
        )
        metrics.update(
            {
                f"{split} accuracy": metric["accuracy"],
                f"{split} macro F1": metric["macro avg"]["f1-score"],
                f"{split} weighted precision": metric["weighted avg"]["precision"],
                f"{split} weighted recall": metric["weighted avg"]["recall"],
            }
        )

    return metrics


site_label_dict_lv2 = {
    "室内-居家住宅": 0,
    "虚拟背景": 1,
    "室内-门店商超": 2,
    "室内-演播室": 3,
    "室内-简易墙面": 4,
    "室内-其他": 5,
    "室内-展示台": 6,
    "室内-仓库工厂": 7,
    "户外-其他": 8,
    "户外-自然户外": 9,
    "室内-公共场所": 10,
    "其他": 11,
    "户外-乡村户外": 12,
    "户外-城市户外": 13,
}


site_label_dict_lv1 = {"室内": 0, "虚拟背景": 1, "户外": 2, "其他": 3}
