# -*- coding: utf-8 -*-
import json
import os
from collections import Counter
from typing import Tuple

from sklearn import metrics


def load_map_dict(symbol_map_path: str, data_format: str = "json") -> Tuple[dict, dict]:
    assert os.path.exists(symbol_map_path)
    if data_format == "json":
        with open(file=symbol_map_path, mode="r", encoding="utf-8") as fr:
            symbol2ids = json.load(fr)
    elif data_format in ("tsv", "csv"):
        sep = "\t" if "tsv" else ","
        symbol2ids = {}
        with open(file=symbol_map_path, mode="r", encoding="utf-8") as fr:
            for i, line in enumerate(fr):
                arr_line = line.strip().split(sep)
                assert len(arr_line) in (1, 2)
                symbol2ids[arr_line[0]] = i
    else:
        raise ValueError(f"Invalid `data_format` value: {data_format}")
    id2symbols = {v: k for (k, v) in symbol2ids.items()}
    return symbol2ids, id2symbols


def get_multi_class_label(label, label2ids):
    if isinstance(label, str):
        label = label.split(",")
    else:
        assert isinstance(label, list)
    multi_hot_label = [0] * len(label2ids)
    for _ in label:
        assert _ in label2ids
        multi_hot_label[label2ids[_]] = 1
    return multi_hot_label


def get_processed_result(y_preds: list, y_trues: list, labels: list):
    processed_y_preds, processed_y_trues = [], []
    label2ids = {k: v for v, k in enumerate(labels)}
    for y_pred, y_true in zip(y_preds, y_trues):
        processed_y_pred = list([_ for _ in y_pred.split(",") if _ in labels])
        processed_y_true = list([_ for _ in y_true.split(",") if _ in labels])

        if len(processed_y_pred) == 0:
            continue
        processed_y_pred = get_multi_class_label(label=processed_y_pred, label2ids=label2ids)
        processed_y_true = get_multi_class_label(label=processed_y_true, label2ids=label2ids)

        processed_y_preds.append(processed_y_pred)
        processed_y_trues.append(processed_y_true)
    return processed_y_preds, processed_y_trues


def evaluate_model(file_path: str, cut: int = 0, ignore_labels: list = None) -> None:
    label_cnts = Counter()
    y_trues, y_preds = [], []

    with open(file_path, mode="r", encoding="utf-8") as fr:
        for line in fr:
            arr_line = line.strip().split("\t")
            assert len(arr_line) in (
                2,
                3,
            ), "File format should be: `text`, `labels`, `predictions(optional)`"
            if len(arr_line) == 2:
                _, labels, predictions = arr_line[0], arr_line[1], neg_label
            else:
                _, labels, predictions = (
                    arr_line[0],
                    arr_line[1],
                    arr_line[2],
                )
            label_cnts.update(labels.split(","))
            y_trues.append(labels)
            y_preds.append(predictions)
    # filter invalid labels
    processed_labels = list([k for k, v in label_cnts.items() if v >= cut and k not in ignore_labels])
    processed_y_preds, processed_y_trues = get_processed_result(
        y_preds=y_preds, y_trues=y_trues, labels=processed_labels
    )
    eval_report = metrics.classification_report(
        y_true=processed_y_trues,
        y_pred=processed_y_preds,
        target_names=processed_labels,
        digits=4,
        output_dict=True,
    )
    print("MICRO-METRIC: {}".format(eval_report["micro avg"]))
    print("MACRO-METRIC: {}".format(eval_report["macro avg"]))


if __name__ == "__main__":
    neg_label = "非负向:非负向:非负向"
    evaluate_model(file_path="eval_detail_example.txt", cut=10, ignore_labels=[neg_label])
