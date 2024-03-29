from typing import Optional

import torch

from .base_running_metric import BaseRunningMetric


class CategoricalAccMetric(BaseRunningMetric):
    """
    Categorical Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    Tie break enables equal distribution of scores among the
    classes with same maximum predicted scores.
    """

    def __init__(self, top_k: int = 1, tie_break: bool = False) -> None:
        if top_k > 1 and tie_break:
            raise Exception("Tie break in Categorical Accuracy can be done only for maximum (top_k = 1)")
        if top_k <= 0:
            raise Exception("top_k passed to Categorical Accuracy must be > 0")
        self._top_k = top_k
        self._tie_break = tie_break
        self.correct_count = 0.0
        self.total_count = 0.0
        self.local_value = 0.0

    def batch_eval(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            predictions : `torch.Tensor`, required.
                A tensor of predictions of shape (batch_size, ..., num_classes).
            gold_labels : `torch.Tensor`, required.
                A tensor of integer class label of shape (batch_size, ...). It must be the same
                shape as the `predictions` tensor without the `num_classes` dimension.
            mask : `torch.Tensor`, optional (default = None).
                A masking tensor the same size as `gold_labels`.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        predictions_within_classes = True
        if predictions.dim() - gold_labels.dim() == 1:
            num_classes = predictions.size(-1)
            if (gold_labels >= num_classes).any():
                raise Exception(
                    "A gold label passed to Categorical Accuracy contains an id >= {}, "
                    "the number of classes.".format(num_classes)
                )
            predictions = predictions.reshape((-1, num_classes)).float()
        else:
            assert self._top_k == 1, "`top_k` should be 1 if `predictions` has no `num_classes` dimension."
            predictions_within_classes = False
            predictions = predictions.reshape(-1).float()

        gold_labels = gold_labels.view(-1).long()
        if not self._tie_break:
            # Top K indexes of the predictions (or fewer, if there aren't K of them).
            # Special case topk == 1, because it's common and .max() is much faster than .topk().
            if self._top_k == 1:
                top_k = (
                    predictions.max(-1)[1].unsqueeze(-1) if predictions_within_classes else predictions.unsqueeze(-1)
                )
            else:
                top_k = predictions.topk(min(self._top_k, predictions.shape[-1]), -1)[1]

            # This is of shape (batch_size, ..., top_k).
            correct = top_k.eq(gold_labels.unsqueeze(-1)).float()
        else:
            # TODO
            assert predictions_within_classes, "`tie_break` requires `predictions` with `num_classes` dimension."

            # prediction is correct if gold label falls on any of the max scores. distribute score by tie_counts
            max_predictions = predictions.max(-1)[0]
            max_predictions_mask = predictions.eq(max_predictions.unsqueeze(-1))
            # max_predictions_mask is (rows X num_classes) and gold_labels is (batch_size)
            # ith entry in gold_labels points to index (0-num_classes) for ith row in max_predictions
            # For each row check if index pointed by gold_label is was 1 or not (among max scored classes)
            correct = max_predictions_mask[
                torch.arange(gold_labels.numel(), device=gold_labels.device).long(),
                gold_labels,
            ].float()
            tie_counts = max_predictions_mask.sum(-1)
            correct /= tie_counts.float()
            correct.unsqueeze_(-1)

        if mask is not None:
            correct *= mask.view(-1, 1).float()
            total_count_add = mask.sum()
        else:
            total_count_add = gold_labels.numel()
        correct_count_add = correct.sum()
        if total_count_add > 1e-12:
            self.local_value = float(correct_count_add) / float(total_count_add)
        else:
            self.local_value = 0.0
        self.total_count += total_count_add
        self.correct_count += correct_count_add

    def get_metric(self, reset: bool = False, use_local: bool = False):
        """
        Returns:
            The accumulated accuracy.
        """
        if not use_local:
            if self.total_count > 1e-12:
                accuracy = float(self.correct_count) / float(self.total_count)
            else:
                accuracy = 0.0
        else:
            accuracy = self.local_value
        if reset:
            self.reset()
        return accuracy

    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0
        self.local_value = 0.0
