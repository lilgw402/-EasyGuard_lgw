import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from cleanlab.count import compute_confident_joint
from cleanlab.filter import find_label_issues
from cruise.utilities.rank_zero import rank_zero_info
from torch import Tensor


class TruncatedLoss(nn.Module):
    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)

    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = ((1 - (Yg**self.q)) / self.q) * self.weight[indexes] - (
            (1 - (self.k**self.q)) / self.q
        ) * self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = (1 - (Yg**self.q)) / self.q
        Lqk = np.repeat(((1 - (self.k**self.q)) / self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)


class MultiFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, gamma=2, reduction="mean"):
        super(MultiFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4

    def forward(self, logit, target):
        num_class = logit.size(-1)
        alpha = (
            torch.ones(
                num_class,
            )
            - 0.5
        )
        if alpha.shape[0] != num_class:
            raise RuntimeError("the length not equal to number of class")

        # filter -1
        indices = target >= 0
        logit = logit[indices]
        target = target[indices]

        # assert isinstance(self.alpha,torch.Tensor)\
        alpha = alpha.to(logit.device)

        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]

        ori_shp = target.shape
        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.squeeze(-1))
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "none":
            loss = loss.view(ori_shp)

        return loss


class InstanceLabelSmoothCrossEntropy(nn.Module):
    def __init__(self, reduction="mean"):
        super(InstanceLabelSmoothCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, pred, smooth_target, label):
        indices = label >= 0
        pred, smooth_target = pred[indices], smooth_target[indices]
        logprobs = F.log_softmax(pred, dim=-1)
        smooth_loss = -(logprobs * smooth_target).sum(dim=-1)
        loss = smooth_loss
        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()


class BatchWeightedCCE(nn.Module):
    """
    Implementing BARE with Cross-Entropy (CCE) Loss
    """

    def __init__(self, k=1, reduction="mean"):
        super(BatchWeightedCCE, self).__init__()

        self.k = k
        self.reduction = reduction

    def forward(self, prediction, target_label, one_hot=True, EPS=1e-8):
        class_num = prediction.shape[1]
        indices = target_label >= 0
        prediction, target_label = prediction[indices], target_label[indices]
        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=class_num).to(target_label.device)
        y_pred = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(y_pred, EPS, 1 - EPS)
        pred_tmp = torch.sum(y_true * y_pred, axis=-1).reshape(-1, 1)

        # Compute batch statistics
        avg_post = torch.mean(y_pred, dim=0)
        avg_post = avg_post.reshape(-1, 1)
        std_post = torch.std(y_pred, dim=0)
        std_post = std_post.reshape(-1, 1)
        avg_post_ref = torch.matmul(y_true.type(torch.float), avg_post)
        std_post_ref = torch.matmul(y_true.type(torch.float), std_post)
        pred_prun = torch.where(
            (pred_tmp - avg_post_ref >= self.k * std_post_ref),
            pred_tmp,
            torch.zeros_like(pred_tmp),
        )

        # prun_idx will tell us which examples are
        # 'trustworthy' for the given batch
        prun_idx = torch.where(pred_prun != 0.0)[0]
        if len(prun_idx) != 0:
            prun_targets = torch.argmax(torch.index_select(y_true, 0, prun_idx), dim=1)
            weighted_loss = F.cross_entropy(
                torch.index_select(prediction, 0, prun_idx),
                prun_targets,
                reduction=self.reduction,
            )
        else:
            weighted_loss = F.cross_entropy(prediction, target_label)
        return weighted_loss


class OnlineConfidentLearning(nn.Module):
    """
    Online version of confident learning
    """

    def __init__(self, n_classes: int, alpha: float = 0.5, smoothing: float = 0.1):
        super(OnlineConfidentLearning, self).__init__()
        assert 0 <= alpha <= 1, "alpha must be in [0, 1]"
        self.a = alpha
        self.n_classes = n_classes
        self.register_buffer("supervise", torch.zeros(n_classes, n_classes))
        self.supervise.fill_(smoothing / (n_classes - 1))
        self.supervise.fill_diagonal_(1 - smoothing)

        self.register_buffer("update", torch.zeros_like(self.supervise))
        self.register_buffer("idx_count", torch.zeros(n_classes))
        self.hard_loss = nn.CrossEntropyLoss()

    def forward(self, y_h: Tensor, y: Tensor):
        # ignore -1 index
        indices = y >= 0
        y_h, y = y_h[indices], y[indices]
        soft_loss = self.soft_loss(y_h, y)
        hard_loss = self.hard_loss(y_h, y)
        return self.a * hard_loss + (1 - self.a) * soft_loss

    def soft_loss(self, y_h: Tensor, y: Tensor):
        y_h = y_h.log_softmax(dim=-1)
        if self.training:
            with torch.no_grad():
                self.step(y_h.exp(), y)
        self.supervise = self.supervise.to(y.device)
        true_dist = torch.index_select(self.supervise, 1, y).swapaxes(-1, -2)
        return torch.mean(torch.sum(-true_dist * y_h, dim=-1))

    def step(self, y_h: Tensor, y: Tensor) -> None:
        y_h_idx = y_h.argmax(dim=-1)
        _, noisy_indexes = compute_confident_joint(
            labels=y.clone().cpu().numpy(),
            pred_probs=y_h.clone().cpu().numpy(),
            return_indices_of_off_diagonals=True,
        )
        mask = torch.ones_like(y, dtype=torch.bool)
        mask[noisy_indexes] = False
        y_h_c = y_h[mask]
        y_h_idx_c = y_h_idx[mask]
        self.update, self.idx_count = self.update.to(y.device), self.idx_count.to(y.device)
        self.update.index_add_(1, y_h_idx_c, y_h_c.swapaxes(-1, -2))
        self.idx_count.index_add_(0, y_h_idx_c, torch.ones_like(y_h_idx_c, dtype=torch.float32))

    def reset(self) -> None:
        self.idx_count[torch.eq(self.idx_count, 0)] = 1
        self.update /= self.idx_count
        self.idx_count.zero_()
        self.supervise = self.update
        self.update = self.update.clone().zero_()


def confident_learning(label_list, logits_list, key_list, fold=0, export_csv=True):
    prob_list = F.softmax(logits_list, dim=-1).tolist()
    prob_list, label_list, key_list = list(
        zip(*[[p, l, k] for p, l, k in zip(prob_list, label_list, key_list) if l >= 0])
    )
    ranked_label_issues = find_label_issues(
        np.array(label_list),
        np.array(prob_list),
        return_indices_ranked_by="self_confidence",
    )
    noisy_keys = [key_list[i] for i in ranked_label_issues]
    noisy_labels = [label_list[i] for i in ranked_label_issues]
    rank_zero_info(f"Cleanlab found {len(ranked_label_issues)} label issues from total {len(label_list)} dataset.")
    rank_zero_info(f"Top 10 most likely label errors: \n {noisy_keys[:10]}")

    if export_csv:
        df = pd.DataFrame(
            list(zip(noisy_keys, noisy_labels)),
            columns=["item_id", "noisy_label"],
        )
        df.to_csv(
            f"examples/video_content_tag/data/output/noisy_keys_fold{fold}.tsv",
            sep="\t",
            index=False,
        )
    return noisy_keys


if __name__ == "__main__":
    k = 3
    b = 4
    criterion = OnlineConfidentLearning(alpha=0.5, n_classes=k, smoothing=0.1)
    logits = torch.randn(b, k).cuda()
    y = torch.randint(k, (b,)).cuda()
    loss = criterion(logits, y)
