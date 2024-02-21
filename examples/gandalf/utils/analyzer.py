# coding=utf-8
import math
from collections import defaultdict

import numpy as np
from utils.driver import get_logger
from utils.file_util import push_files

MIN_SAMPLE_COUNT = 1000
MIN_POSITIVE_SAMPLE_COUNT = 100


def explained_variance_score_sklearn(preds, labels, multioutput="uniform_average"):
    """
    越接近1效果越好
    """
    from sklearn.metrics import explained_variance_score

    evs = explained_variance_score(labels, preds, multioutput=multioutput)
    return evs


def mae_sklearn(preds, labels, multioutput="uniform_average"):
    from sklearn.metrics import mean_absolute_error

    mae = mean_absolute_error(labels, preds, multioutput=multioutput)
    return mae


def mse_sklearn(preds, labels, multioutput="uniform_average"):
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(labels, preds, multioutput=multioutput)
    return mse


def rmse_sklearn(preds, labels, multioutput="uniform_average"):
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(labels, preds, multioutput=multioutput)
    rmse = np.sqrt(mse)
    return rmse


def r2_score(preds, labels, multioutput="uniform_average"):
    from sklearn.metrics import r2_score

    r2s = r2_score(labels, preds, multioutput=multioutput)
    return r2s


def mape(preds, labels):
    return np.mean(np.abs((preds - labels) / labels)) * 100


def smape(preds, labels):
    return 2.0 * np.mean(np.abs(preds - labels) / (np.abs(preds) + np.abs(labels))) * 100


def roc_curve_sklearn(preds, labels):
    from sklearn.metrics import auc, roc_curve

    if len(labels) < MIN_SAMPLE_COUNT or sum(labels) < MIN_POSITIVE_SAMPLE_COUNT:
        return 0.0
    fpr, tpr, threshold = roc_curve(labels, preds, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def accuracy_score(preds, labels, threshold=0.5):
    import numpy as np
    from sklearn.metrics import accuracy_score

    pred_labels = np.array([1 if pred > threshold else 0 for pred in preds])
    return accuracy_score(labels, pred_labels)


def precision_rescall_score(preds, labels, threshold=0.5):
    """仅适用于二分类"""
    import numpy as np

    total_count = len(preds)
    inter_review_positive_count = np.sum(
        np.array([1 if preds[i] > threshold and labels[i] == 1 else 0 for i in range(total_count)])
    )
    inter_review_negative_count = np.sum(
        np.array([1 if preds[i] > threshold and labels[i] == 0 else 0 for i in range(total_count)])
    )
    inter_review_count = np.sum(np.array([1 if pred > threshold else 0 for pred in preds])) + 1e-20
    total_positive_count = np.sum(labels) + 1e-20

    review_rate = 1.0 * inter_review_count / (total_count + 1e-5)  # 进审率（影响面），预估大于阈值的样本量/样本总量
    review_detail = "({:.0f}/{:.0f})".format(inter_review_count, total_count)
    precision = 1.0 * inter_review_positive_count / (inter_review_count + 1e-5)  # 审出率，进审的正样本量/进审样本量
    precision_detail = "({:.0f}/{:.0f})".format(inter_review_positive_count, inter_review_count)
    recall = 1.0 * inter_review_positive_count / (total_positive_count + 1e-5)  # 召回率，进审的正样本量/正样本量
    recall_detail = "({:.0f}/{:.0f})".format(inter_review_positive_count, total_positive_count)
    f1_measure = 2.0 * ((precision * recall) / (precision + recall + 1e-13))
    f1_detail = "({}/{})".format("", "")
    # 漏放率
    leak_rate = (1 - recall) * total_positive_count / ((1 - review_rate) * len(labels))
    leak_detail = "({:.0f}/{:.0f})".format((1 - recall) * total_positive_count, ((1 - review_rate) * len(labels)))
    # 误伤率
    err_rate = inter_review_negative_count / (inter_review_negative_count + inter_review_positive_count + 1e-5)
    err_detail = "({:.0f}/{:.0f})".format(
        inter_review_negative_count,
        inter_review_negative_count + inter_review_positive_count,
    )
    detail_set = [
        precision_detail,
        recall_detail,
        review_detail,
        leak_detail,
        err_detail,
        f1_detail,
    ]
    return (
        precision,
        recall,
        review_rate,
        leak_rate,
        err_rate,
        f1_measure,
        detail_set,
    )


def current_score(labels, review):
    import numpy as np

    lens = len(labels)
    presum = np.sum(labels) + 1e-20
    presum_review = np.sum(review) + 1e-20

    review_rate = 1.0 * presum_review / lens
    precision_rate = 1.0 * presum / presum_review

    return review_rate, precision_rate


# The optimiztion object of logistic regression
def log_loss_sklearn(preds, labels):
    from sklearn.metrics import log_loss

    return log_loss(preds, labels)


# The optimiztion object of logistic regression
def log_loss(preds, labels):
    if len(preds) < MIN_POSITIVE_SAMPLE_COUNT:
        return None
    total_error = 0.0
    for i in range(len(preds)):
        try:
            preds[i] = 0.9999 if preds[i] == 1.0 else preds[i]
            preds[i] = 0.00001 if preds[i] == 0.0 else preds[i]
            total_error += labels[i] * math.log(preds[i]) + (1.0 - labels[i]) * math.log(1.0 - preds[i])
        except Exception as e:
            get_logger().info("error: %s, number: %s, pred: %s" % (e, i, preds[i]))
    loss = total_error / (-float(len(preds)))
    return loss


def trapezoid_area(fp, pre_fp, tp, pre_tp):
    base = abs(fp - pre_fp)
    height = (tp + pre_tp) / 2
    return base * height


def roc_curve_custom(preds, labels):
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    labels = [True if labels[i] > 0 else False for i in range(len(labels))]
    preds_prob = {}
    labels_dict = {}
    for i, prob in enumerate(preds):
        preds_prob[i] = prob
    for i, label in enumerate(labels):
        labels_dict[i] = label

    pred_items = preds_prob.items()
    pred_sorted_items = [[v[1], v[0]] for v in pred_items]
    pred_sorted_items.sort(reverse=True)

    true_positive_current_count = 0.0
    false_positive_current_count = 0.0
    previous_true_positive_current_count = 0.0
    previous_false_positive_current_count = 0.0

    area_under_curve = 0.0
    previous_prediction_value = 0.0
    for item in pred_sorted_items:
        if not previous_prediction_value == item[0]:
            area_under_curve += trapezoid_area(
                false_positive_current_count,
                previous_false_positive_current_count,
                true_positive_current_count,
                previous_true_positive_current_count,
            )
            previous_prediction_value = item[0]
            previous_true_positive_current_count = true_positive_current_count
            previous_false_positive_current_count = false_positive_current_count

        if labels_dict[item[1]]:
            true_positive_current_count += 1
        else:
            false_positive_current_count += 1

    area_under_curve += trapezoid_area(
        false_positive_current_count,
        previous_false_positive_current_count,
        true_positive_current_count,
        previous_true_positive_current_count,
    )

    auc = area_under_curve / ((positive_count * negative_count) + 1e-20)
    return auc


def pr_score_analysis(
    preds,
    labels,
    start=0,
    end=1,
    partition=250,
    fixed_recall=0.3,
    dump_table_name=None,
    hdfs_table_name=None,
):
    table = defaultdict(list)
    fix_recall_dict = fix_recall_analysis(preds, labels, fixed_recall=fixed_recall)
    table["阈值"].append(fix_recall_dict["threshold"])
    table["进审率"].append(fix_recall_dict["review_rate"])
    table["审出率"].append(fix_recall_dict["precision"])
    table["召回率"].append(fix_recall_dict["recall"])
    table["漏放率"].append(fix_recall_dict["leak_rate"])
    table["误伤率"].append(fix_recall_dict["err_rate"])
    table["acc"].append(fix_recall_dict["accuracy"])
    table["f1"].append(fix_recall_dict["f1"])

    for i in range(partition):
        threshold = start + (end - start) * i / partition
        (
            precision,
            recall,
            review_rate,
            leak_rate,
            err_rate,
            f1,
            details,
        ) = precision_rescall_score(preds, labels, threshold=threshold)
        (
            precision_detail,
            recall_detail,
            review_detail,
            leak_detail,
            err_detail,
            f1_detail,
        ) = details
        accuracy = accuracy_score(preds, labels, threshold=threshold)
        get_logger().info(
            f"阈值: {threshold:.3f}, "
            f"进审率: {review_rate:.2%}{review_detail}, "
            f"审出率: {precision:.2%}{precision_detail}, "
            f"召回率: {recall:.2%}{recall_detail}, "
            f"漏放率: {leak_rate:.2%}{leak_detail}, "
            f"误伤率: {err_rate:.2%}{err_detail}, "
            f"acc: {accuracy:.4f}, "
            f"f1: {f1:.4f}"
        )
        if dump_table_name:
            table["阈值"].append(threshold)
            table["进审率"].append(review_rate)
            table["审出率"].append(precision)
            table["召回率"].append(recall)
            table["漏放率"].append(leak_rate)
            table["误伤率"].append(err_rate)
            table["acc"].append(accuracy)
            table["f1"].append(f1)

    if dump_table_name and hdfs_table_name:
        import pandas as pd

        pd.DataFrame.from_dict(table).to_csv(dump_table_name, index=False)
        push_files(dump_table_name, hdfs_table_name)
    return fix_recall_dict


def get_threshold_for_recall(preds, labels, fixed_recall=0.3):
    threshold = 0.5
    target = int(math.ceil(sum(labels) * fixed_recall))
    assert len(preds) == len(labels)
    sorted_pairs = sorted(zip(preds, labels), key=lambda pair: pair[0], reverse=True)
    count = 0
    for idx in range(len(preds)):
        if int(sorted_pairs[idx][1]) == 1:
            count += 1
        if count == target:
            threshold = sorted_pairs[idx][0]
            break
    return threshold


def fix_recall_analysis(preds, labels, fixed_recall=0.3):
    threshold = get_threshold_for_recall(preds, labels, fixed_recall)
    (
        precision,
        recall,
        review_rate,
        leak_rate,
        err_rate,
        f1,
        details,
    ) = precision_rescall_score(preds, labels, threshold=threshold)
    (
        precision_detail,
        recall_detail,
        review_detail,
        leak_detail,
        err_detail,
        f1_detail,
    ) = details
    accuracy = accuracy_score(preds, labels, threshold=threshold)
    get_logger().info(
        f"阈值: {threshold:.3f}, "
        f"进审率: {review_rate:.2%}{review_detail}, "
        f"审出率: {precision:.2%}{precision_detail}, "
        f"召回率: {recall:.2%}{recall_detail}, "
        f"漏放率: {leak_rate:.2%}{leak_detail}, "
        f"误伤率: {err_rate:.2%}{err_detail}, "
        f"acc: {accuracy:.4f}, "
        f"f1: {f1:.4f}"
    )
    return {
        "threshold": f"{threshold:.4f}",
        "review_rate": f"{review_rate:.2%}",
        "precision": f"{precision:.2%}",
        "recall": f"{recall:.2%}",
        "leak_rate": f"{leak_rate:.2%}",
        "err_rate": f"{err_rate:.2%}",
        "accuracy": f"{accuracy:.2%}",
        "f1": f"{f1:.4f}",
    }
