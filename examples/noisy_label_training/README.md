# This is an example of nosiy label training.

## mixgen
1. 代码实现，比较简单，只需混合数据，不需要混合标签，即插即用。
```
def mixgen(data, lam=0.5):
    batch_size = len(data) // 4
    index = np.random.permutation(batch_size)
    for i in range(batch_size):
        # image mixup
        for j in range(len(data[i]['frames'])):
            data[i]['frames'][j] = lam * data[i]['frames'][j] + (1 - lam) * data[index[i]]['frames'][j]
        # text concat
        data[i]['input_ids'] = data[i]['input_ids'] + data[index[i]]['input_ids'] 
    return data
```
2. 训练脚本
```
./tools/TORCHRUN examples/noisy_label_training/run_mixgen.py \
    --config examples/noisy_label_training/configs/config_mixgen.yaml
```

## confident learning
1. 需要进行交叉验证，一般将训练集分成三个fold，交叉验证训练
2. 每次训练结束后调用下面的confident_learning方法，得到每个fold的噪声样本
3. 对噪声样本进行剔除或重标，再用剔除噪声或重标后的训练集进行训练
```
def confident_learning(label_list, logits_list, key_list, fold=0, export_csv=True):
    prob_list = F.softmax(logits_list, dim=-1).tolist()
    prob_list, label_list, key_list = list(
        zip(*[[p, l, k] for p, l, k in zip(prob_list, label_list, key_list) if l >= 0]))
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
        df = pd.DataFrame(list(zip(noisy_keys, noisy_labels)), columns=['item_id', 'noisy_label'])
        df.to_csv(f"examples/video_content_tag/data/output/noisy_keys_fold{fold}.tsv", sep='\t', index=False)
    return noisy_keys
```

## online confident learning
1. 不需要交叉验证，只需改变loss function
```
class OnlineConfidentLearning(nn.Module):
    """
    Online version of confident learning
    """

    def __init__(self, n_classes: int, alpha: float = 0.5, smoothing: float = 0.1):
        super(OnlineConfidentLearning, self).__init__()
        assert 0 <= alpha <= 1, 'alpha must be in [0, 1]'
        self.a = alpha
        self.n_classes = n_classes
        self.register_buffer('supervise', torch.zeros(n_classes, n_classes))
        self.supervise.fill_(smoothing / (n_classes - 1))
        self.supervise.fill_diagonal_(1 - smoothing)

        self.register_buffer('update', torch.zeros_like(self.supervise))
        self.register_buffer('idx_count', torch.zeros(n_classes))
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
```
2. 训练流程中加入一行reset代码
```
def on_train_epoch_end(self):
    self.criterion.reset()
```
3. 训练脚本
```
./tools/TORCHRUN examples/noisy_label_training/run_ocl.py \
    --config examples/noisy_label_training/configs/config_ocl.yaml
```