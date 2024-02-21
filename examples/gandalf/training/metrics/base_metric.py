import json
import re


class BaseMetric:
    name = None

    def cal(self, file_path_list):
        result = self.load_json_result(file_path_list)
        scores, labels = self.collect_pre_labels(result)
        metrics = self.cal_metric(scores, labels)
        return metrics

    def cal_metric(self, scores, labels) -> dict:
        raise NotImplementedError

    def collect_pre_labels(self, result):
        raise NotImplementedError

    def load_json_result(self, json_path_list):
        total_result = []
        if "rank" in json_path_list[0]:
            regex = r"(.*?)_rank"
        else:
            regex = r"(.*?)\."
        self.name = "_".join(re.findall(regex, json_path_list[0].split("/")[-1]))
        for json_path in json_path_list:
            with open(json_path) as f:
                total_result.extend(json.load(f))
        return total_result


class Dumper:
    name = None

    def cal(self, file_path_list):
        result = self.load_json_result(file_path_list)
        self.dump(result)

    def dump(self, result) -> dict:
        raise NotImplementedError

    def load_json_result(self, json_path_list):
        total_result = []
        if not json_path_list:
            return
        if "rank" in json_path_list[0]:
            regex = r"(.*?)_rank"
        else:
            regex = r"(.*?)\."
        self.name = "_".join(re.findall(regex, json_path_list[0].split("/")[-1]))
        for json_path in json_path_list:
            with open(json_path) as f:
                total_result.extend(json.load(f))
        return total_result
