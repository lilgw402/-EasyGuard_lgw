# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2022-11-10 11:04:47
# Modified: 2022-11-10 11:04:47
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from tabulate import tabulate
from training.metrics.base_metric import Dumper
from training.metrics.simple_metrics import precision_rescall_score
from utils.registry import METRICS


@METRICS.register_module()
class GandalfDumper(Dumper):
    def __init__(self, collect_fields, save_path, strict=True, **kwargs):
        self.collect_fields = collect_fields
        assert self.collect_fields, "empty fields found!"
        self.save_path = save_path
        self.strict = strict
        self.df = None
        self.output_field = "output"
        self.label_filed = kwargs.get("label_filed", "label")
        self.tag_field = kwargs.get("tag_field", "verify_reason")
        self.old_score_filed = kwargs.get("old_score_filed", "online_score")
        self.old_high_threshold = kwargs.get("old_high_threshold", 0.164)
        self.old_low_threshold = kwargs.get("old_low_threshold", 0.499)

    def dump(self, records) -> None:
        table = defaultdict(list)
        self.save_path = self.save_path.replace(".csv", f"_{self.name}.csv")
        for record in records:
            for key in self.collect_fields:
                if key not in record:
                    raise KeyError(f"missing key: {key} in record")
                table[key].append(record[key])
        pd.DataFrame.from_dict(table).to_csv(self.save_path, index=False)
        self.cal_label_metric(split_uv=True)

    @staticmethod
    def get_threshold_by_order(
        df: pd.DataFrame,
        recall: int,
        score_col: str,
        label_col: str,
        quantile=None,
    ) -> float:
        if quantile:
            recall = np.ceil(df[df[label_col] == 1].shape[0] * quantile)
        tp, threshold, hit = 0, 0.0, 0
        df = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
        for idx in df.index:
            hit += 1
            label = df.loc[idx, label_col]
            if int(label) == 1:
                tp += 1
            if tp == recall:
                threshold = df.loc[idx, score_col]
                break
        if tp < recall:
            print("Warning:Recalled less than recall!!!")
        return threshold, hit

    def get_label_metric(self, df, threshold=None, score_col="output", label_col="label"):
        tag_metric = defaultdict(list)
        for name, sub_df in df.groupby(by=self.tag_field):
            (
                precision,
                recall,
                review_rate,
                leak_rate,
                err_rate,
                f1_measure,
                detail_set,
            ) = precision_rescall_score(
                sub_df.loc[:, score_col].tolist(),
                sub_df.loc[:, label_col].tolist(),
                threshold=threshold,
            )
            print(f"tag:{name} low p:{precision} r:{recall}, f1:{f1_measure}")
            tag_metric["name"].append(name)
            tag_metric["support"].append(sub_df.loc[:, "label"].sum())
            tag_metric["precision"].append(f"{precision:.3%}")
            tag_metric["recall"].append(f"{recall:.3%}")
        return pd.DataFrame.from_dict(tag_metric).sort_values(["name", "support"], ascending=[True, True])

    def get_support(self, split_uv=True):
        df = pd.read_csv(self.save_path)
        if not split_uv:
            self.support = df[df[self.label_filed] == 1].shape[0]
            self.old_hit = df[df[self.old_score_filed] >= self.old_low_threshold].shape[0]
            self.old_recall = df[
                (df[self.label_filed] == 1) & (df[self.old_score_filed] >= self.old_low_threshold)
            ].shape[0]
            return df
        else:
            high_df = df[(df.uv >= 100)]
            low_df = df[(df.uv < 100)]
            self.high_support = high_df[high_df[self.label_filed] == 1].shape[
                0
            ]  # Consider the timeseries exempt, this label is not accurate
            self.low_support = low_df[low_df[self.label_filed] == 1].shape[0]
            self.old_high_recall = high_df[
                (high_df[self.label_filed] == 1) & (high_df[self.old_score_filed] >= self.old_high_threshold)
            ].shape[0]
            self.old_low_recall = low_df[
                (low_df[self.label_filed] == 1) & (low_df[self.old_score_filed] >= self.old_low_threshold)
            ].shape[0]
            self.old_high_hit = high_df[high_df[self.old_score_filed] >= self.old_high_threshold].shape[0]
            self.old_low_hit = low_df[low_df[self.old_score_filed] >= self.old_low_threshold].shape[0]
            return high_df, low_df

    def cal_threshold(self, split_uv=True):
        if not split_uv:
            df = self.get_support(split_uv)
            threshold, hit = self.get_threshold_by_order(
                df,
                recall=self.recall,
                score_col=self.output_field,
                label_col=self.label_filed,
            )
            logging.info(
                f"Total:{df.shape[0]},\n"
                f"Support:{self.support},\n"
                f"Recall:{self.old_recall},\n"
                f"Old Hit:{self.old_hit},\n"
                f"Old Censor ratio:{self.old_hit/df.shape[0]:.2%},\n"
                f"Threshold:{threshold:5f}"
                f"Hit:{hit},\n"
                f"Censor ratio:{hit/df.shape[0]:.2%},\n"
                f"Threshold:{threshold:5f}"
            )
            return df, threshold
        else:
            high_df, low_df = self.get_support(split_uv)
            high_threshold, high_hit = self.get_threshold_by_order(
                high_df,
                recall=self.old_high_recall,
                score_col=self.output_field,
                label_col=self.label_filed,
            )
            low_threshold, low_hit = self.get_threshold_by_order(
                low_df,
                recall=self.old_low_recall,
                score_col=self.output_field,
                label_col=self.label_filed,
            )
            logging.info(
                f"High total:{high_df.shape[0]},\n"
                f"High support:{self.high_support},\n"
                f"High recall:{self.old_high_recall},\n"
                f"Old high hit:{self.old_high_hit},\n"
                f"Old high censor ratio:{self.old_high_hit/high_df.shape[0]:.2%},\n"
                f"High hit:{high_hit},\n"
                f"High censor ratio:{high_hit/high_df.shape[0]:.2%},\n"
                f"High Threshold:{high_threshold:5f},\n"
                f"Low total:{low_df.shape[0]},\n"
                f"Low support:{self.low_support},\n"
                f"Low recall:{self.old_low_recall},\n"
                f"Old low hit:{self.old_low_hit},\n"
                f"Old Censor ratio:{self.old_low_hit/low_df.shape[0]:.2%},\n"
                f"Low hit:{low_hit},\n"
                f"Low censor ratio:{low_hit/low_df.shape[0]:.2%},\n"
                f"Low Threshold:{low_threshold:5f}"
            )
            return high_df, high_threshold, low_df, low_threshold

    def cal_label_metric(self, split_uv=True):
        if not split_uv:
            print("====Cal old metric====")
            old_tag_metric = self.get_label_metric(
                self.df,
                self.old_high_threshold,
                score_col=self.old_score_filed,
                label_col=self.label_filed,
            )
            print(
                tabulate(
                    old_tag_metric,
                    headers="keys",
                    tablefmt="simple_grid",
                    colalign=("central",),
                )
            )
            print("====Cal new metric====")
            df, threshold = self.cal_threshold(split_uv)
            tag_metric = self.get_label_metric(
                df,
                threshold,
                score_col=self.output_field,
                label_col=self.label_filed,
            )
            print("====Show diff metric====")
            diff_metric = pd.merge(old_tag_metric, tag_metric, how="left", on=["name", "support"])
            print(
                tabulate(
                    diff_metric,
                    headers="keys",
                    tablefmt="simple_grid",
                    colalign=("central",),
                )
            )
        else:
            (
                high_df,
                high_threshold,
                low_df,
                low_threshold,
            ) = self.cal_threshold(split_uv)
            print("====Cal old high uv metric====")
            old_high_tag_metric = self.get_label_metric(
                high_df,
                self.old_high_threshold,
                score_col=self.old_score_filed,
                label_col=self.label_filed,
            )
            print("====Cal new high uv metric====")
            high_tag_metric = self.get_label_metric(
                high_df,
                high_threshold,
                score_col=self.output_field,
                label_col=self.label_filed,
            )
            print("====Show high uv diff metric====")
            high_diff_metric = pd.merge(
                old_high_tag_metric,
                high_tag_metric,
                how="left",
                on=["name", "support"],
            )
            print(
                tabulate(
                    high_diff_metric,
                    headers="keys",
                    tablefmt="simple_grid",
                    colalign=("central",),
                )
            )
            # old_recall, new_recall = 0, 0
            # for idx in high_diff_metric.index:
            # 	old_recall += float(high_diff_metric.loc[idx,'recall_x'].strip('%'))/100*high_diff_metric.loc[idx,'support']
            # 	new_recall += float(high_diff_metric.loc[idx,'recall_y'].strip('%'))/100*high_diff_metric.loc[idx,'support']
            # print(old_recall,new_recall)
            print("====Cal old low uv metric====")
            old_low_tag_metric = self.get_label_metric(
                low_df,
                self.old_low_threshold,
                score_col=self.old_score_filed,
                label_col=self.label_filed,
            )
            print("====Cal new low uv metric====")
            low_tag_metric = self.get_label_metric(
                low_df,
                low_threshold,
                score_col=self.output_field,
                label_col=self.label_filed,
            )
            print("====Show low uv diff metric====")
            low_diff_metric = pd.merge(
                old_low_tag_metric,
                low_tag_metric,
                how="left",
                on=["name", "support"],
            )
            print(
                tabulate(
                    low_diff_metric,
                    headers="keys",
                    tablefmt="simple_grid",
                    colalign=("central",),
                )
            )
            # old_recall, new_recall = 0, 0
            # for idx in low_diff_metric.index:
            # 	old_recall += float(low_diff_metric.loc[idx,'recall_x'].strip('%'))/100*low_diff_metric.loc[idx,'support']
            # 	new_recall += float(low_diff_metric.loc[idx,'recall_y'].strip('%'))/100*low_diff_metric.loc[idx,'support']
            # print(old_recall,new_recall)


if __name__ == "__main__":
    dumper = GandalfDumper(
        [
            "object_id",
            "verify_reason",
            "uv",
            "online_score",
            "output",
            "label",
        ],
        "/mlx_devbox/users/jiangxubin/repo/121/maxwell/id2score_2022102-7-9--2022103-0-1--2022110-1-2-.csv",
        strict=True,
    )
    dumper.cal_label_metric()
