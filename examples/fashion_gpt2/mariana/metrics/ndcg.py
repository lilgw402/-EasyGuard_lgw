import numpy as np


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.0


def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.0
    return dcg_at_k(r, k) / idcg


def calc_ndcg(labels, scores):
    indexes = [i for i in range(len(scores))]
    new_rank = sorted(zip(scores, indexes), key=lambda x: x[0], reverse=True)
    new_labels = []
    for _, idx in new_rank:
        new_labels.append(labels[idx])
    ndcg1 = ndcg_at_k(new_labels, 1)
    ndcg3 = ndcg_at_k(new_labels, 3)
    ndcg5 = ndcg_at_k(new_labels, 5)
    return ndcg1, ndcg3, ndcg5


def evaluate_ndcg(labels, predicts):
    ndcg1_records, ndcg3_records, ndcg5_records = [], [], []
    for label, predict in zip(labels, predicts):
        ndcg1, ndcg3, ndcg5 = calc_ndcg(label, predict)
        ndcg1_records.append(ndcg1)
        ndcg3_records.append(ndcg3)
        ndcg5_records.append(ndcg5)
    result = {
        "ndcg1": sum(ndcg1_records) / len(ndcg1_records),
        "ndcg3": sum(ndcg3_records) / len(ndcg3_records),
        "ndcg5": sum(ndcg5_records) / len(ndcg5_records),
    }
    return result
