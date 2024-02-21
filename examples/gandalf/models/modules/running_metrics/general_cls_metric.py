import numpy as np
import torch
from sklearn.metrics import auc, precision_score, recall_score, roc_curve

from .base_running_metric import BaseRunningMetric


class GeneralClsMetric(BaseRunningMetric):
    def __init__(self, binary_threshold=0.5, multi_class=False, neg_tag=0):
        self.binary_threshold = binary_threshold
        self.multi_class = multi_class
        self.neg_tag = neg_tag
        assert self.neg_tag != 1, "neg_tag=1 is ambiguous, choose another num"

    def batch_eval(self, output_tensor, targets_tensor, key=""):
        use_cuda = output_tensor.is_cuda
        original_device = output_tensor.device
        if use_cuda:
            output_arr = output_tensor.detach().cpu().numpy()
            targets_arr = targets_tensor.detach().cpu().numpy()
        else:
            output_arr = output_tensor.detach().numpy()
            targets_arr = targets_tensor.detach().numpy()

        output_arr = np.nan_to_num(output_arr)

        if self.multi_class:
            output_arr = np.argmax(output_arr, axis=1)
        else:
            output_arr[output_arr > self.binary_threshold] = 1
            output_arr[output_arr <= self.binary_threshold] = self.neg_tag
        input_neg_ratio = np.sum(targets_arr == self.neg_tag) / targets_arr.shape[0]
        input_pos_ratio = 1 - input_neg_ratio
        output_neg_ratio = np.sum(output_arr == self.neg_tag) / output_arr.shape[0]
        output_pos_ratio = 1 - output_neg_ratio

        acc = np.sum(targets_arr == output_arr) / targets_arr.shape[0]
        binary_output = np.array(output_arr != self.neg_tag, dtype=np.int64)
        binary_labels = np.array(targets_arr != self.neg_tag, dtype=np.int64)
        binary_prec = precision_score(binary_labels, binary_output, zero_division=0)
        binary_recall = recall_score(binary_labels, binary_output, zero_division=0)
        binary_fpr, binary_tpr, _ = roc_curve(binary_labels, binary_output, pos_label=1)
        binary_auc = auc(binary_fpr, binary_tpr)
        if binary_auc != binary_auc:
            binary_auc = 0
        binary_f1 = 2 * (binary_prec * binary_recall) / (binary_prec + binary_recall + 1e-6)

        def torch_wrapper(raw_scalar):
            if use_cuda:
                return torch.tensor(np.full_like(targets_arr, fill_value=raw_scalar, dtype=np.float32)).to(
                    original_device
                )
            else:
                return torch.tensor(np.full_like(targets_arr, fill_value=raw_scalar, dtype=np.float32))

        key = key + "_" if key else key
        return {
            f"{key}acc": acc,
            f"{key}binary_prec": binary_prec,
            f"{key}binary_recall": binary_recall,
            f"{key}binary_f1": binary_f1,
            f"{key}input_neg_ratio": input_neg_ratio,
            f"{key}input_pos_ratio": input_pos_ratio,
            f"{key}output_neg_ratio": output_neg_ratio,
            f"{key}output_pos_ratio": output_pos_ratio,
            f"{key}binary_auc": binary_auc,
        }
        # return {
        #     f"{key}acc": torch_wrapper(acc),
        #     f"{key}binary_prec": torch_wrapper(binary_prec),
        #     f"{key}binary_recall": torch_wrapper(binary_recall),
        #     f"{key}binary_f1": torch_wrapper(binary_f1),
        #     f"{key}input_neg_ratio": torch_wrapper(input_neg_ratio),
        #     f"{key}input_pos_ratio": torch_wrapper(input_pos_ratio),
        #     f"{key}output_neg_ratio": torch_wrapper(output_neg_ratio),
        #     f"{key}output_pos_ratio": torch_wrapper(output_pos_ratio),
        #     f"{key}binary_auc": torch_wrapper(binary_auc),
        # }


if __name__ == "__main__":
    outputs = torch.rand([4, 1]) * 0.47
    targets = torch.tensor([0, 0, 0, 0])
    metric = GeneralClsMetric()
    print(metric.batch_eval(outputs, targets))
