import os
import sys

import numpy as np

try:
    import easyguard
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cruise import CruiseCLI, CruiseTrainer
from sklearn.metrics import roc_auc_score

from easyguard.appzoo.high_quality_live.data import HighQualityLiveDataModule
from easyguard.appzoo.high_quality_live.model_videoclip import HighQualityLiveVideoCLIP


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
    precision, recall, thr = p_fix_r(prob, labels, 0.3)
    print(str(precision * 100)[:5] + " / " + str(recall * 100)[:5] + " / " + str(thr)[:5])
    precision, recall, thr = r_fix_p(prob, labels, 0.5)
    print(str(precision * 100)[:5] + " / " + str(recall * 100)[:5] + " / " + str(thr)[:5])


cli = CruiseCLI(
    HighQualityLiveVideoCLIP,
    trainer_class=CruiseTrainer,
    datamodule_class=HighQualityLiveDataModule,
    trainer_defaults={
        "log_every_n_steps": 50,
        "precision": "fp16",
        "sync_batchnorm": True,
        "find_unused_parameters": True,
        "summarize_model_depth": 2,
    },
)
# pdb.set_trace()
cfg, trainer, model, datamodule = cli.parse_args()
# pdb.set_trace()
# 预测得分
outputs = trainer.predict(
    model,
    predict_dataloader=datamodule.predict_dataloader(
        data_source="/mnt/bn/ecom-govern-maxiangqian/liuzeyu/hq_live/packed_data/20221103_20221106/v1_test/test2w",
        batch_size=32,
        num_workers=8,
    ),
    sync_predictions=True,
)

# print(outputs)

# dump predictions
# save_dir = '/mnt/bn/ecom-govern-maxiangqian/liuzeyu/authentic/output_results'
# save_dir = '/mnt/bn/ecom-govern-maxiangqian/liuzeyu/hq_live/output_results'
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# with open(os.path.join(save_dir, 'result-test2w-v1_train16w_balance_1109-pass26.txt'), 'wt') as fw:
#     for out in outputs:
#         pred = out['pred'].cpu().numpy()
#         grt = out['label'].cpu().numpy()
#         item_ids = out['item_id']
#         for i in range(pred.shape[0]):
#             fw.write('{} {} {} {} {}\n'.format(item_ids[i], pred[i, 0], pred[i, 1], pred[i, 2], grt[i]))
label = []
score = []
for out in outputs:
    pred = out["pred"].cpu().numpy()
    grt = out["label"].cpu().numpy()
    for i in range(pred.shape[0]):
        score.append(pred[i, 2])
        if int(grt[i]) == 2:
            label.append(1)
        else:
            label.append(0)
print(len(label), len(score), sum(label))
print("auc: ", roc_auc_score(label, score))
print_res(np.array(score), np.array(label))
