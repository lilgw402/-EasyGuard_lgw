import torch
from cruise import CruiseModule
from torch import nn
from transformers import AutoModel

from easyguard.core.optimizers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from easyguard.modelzoo.models.fashionproduct_xl.albert import Transformer
from easyguard.modelzoo.models.fashionproduct_xl.swin import SwinTransformer


class FashionBertXL(CruiseModule):
    def __init__(
        self,
        # backbone='fashionproduct-xl-general-v1',
        class_num: int = 1900,
        hidden_dim: int = 768,
        head_num: int = 5,
        use_multihead: bool = True,
        learning_rate: float = 1.0e-4,
        weight_decay: float = 1.0e-4,
        lr_schedule: str = "linear",
        warmup_steps_factor: float = 4,
        eps: float = 1e-8,
        embd_pdrop: float = 0.1,
        optim: str = "AdamW",
        low_lr_prefix: list = [],
        freeze_prefix: list = [],
        load_pretrained: str = None,
        prefix_changes: list = [],
        download_files: list = [],
    ):
        super(FashionBertXL, self).__init__()
        self.save_hparams()

    def setup(self, stage) -> None:
        self.text = AutoModel.from_pretrained("/opt/tiger/xlm-roberta-base-torch")
        self.visual = SwinTransformer(
            img_size=224,
            num_classes=self.hparams.hidden_dim,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
        )
        from types import SimpleNamespace

        cls_config_dict = {
            "num_hidden_layers": 4,
            "hidden_size": self.hparams.hidden_dim,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "hidden_dropout_prob": 0.1,
            "layernorm_eps": 1e-05,
        }
        cls_config = SimpleNamespace(**cls_config_dict)
        self.fuse = Transformer(cls_config)

        self.pos_embedding = nn.Embedding(528, self.hparams.hidden_dim)  # t, v embeddiong
        self.v_projector = nn.Sequential(
            nn.Linear(self.hparams.hidden_dim, 2 * self.hparams.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * self.hparams.hidden_dim, self.hparams.hidden_dim),
        )

        self.t_header = nn.Sequential(
            # nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
            nn.Dropout(0.1),
            # nn.GELU(),
            nn.Linear(self.hparams.hidden_dim, self.hparams.class_num),
        )
        self.v_header = nn.Sequential(
            # nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
            nn.Dropout(0.1),
            # nn.GELU(),
            nn.Linear(self.hparams.hidden_dim, self.hparams.class_num),
        )
        self.f_header = nn.Sequential(
            # nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
            nn.Dropout(0.1),
            # nn.GELU(),
            nn.Linear(self.hparams.hidden_dim, self.hparams.class_num),
        )
        if self.hparams.use_multihead:
            self.multi_heads = torch.nn.Linear(self.hparams.hidden_dim, self.hparams.head_num * self.hparams.class_num)
        else:
            self.f_cls = nn.Linear(self.hparams.hidden_dim, self.hparams.class_num)

        self.ce = nn.CrossEntropyLoss()

        if self.hparams.load_pretrained:
            prefix_changes = [prefix_change.split("->") for prefix_change in self.hparams.prefix_changes]
            rename_params = {pretrain_prefix: new_prefix for pretrain_prefix, new_prefix in prefix_changes}
            self.partial_load_from_checkpoints(
                self.hparams.load_pretrained, map_location="cpu", rename_params=rename_params
            )
        self.freeze_params(self.hparams.freeze_prefix)

    def local_rank_zero_prepare(self) -> None:
        import os

        if self.hparams.download_files:
            to_download = [df.split("->") for df in self.hparams.download_files]
            for src, tar in to_download:
                if not os.path.exists(tar):
                    os.makedirs(tar)
                fdname = src.split("/")[-1]
                if os.path.exists(f"{tar}/{fdname}"):
                    print(f"{tar}/{fdname} already existed, pass!")
                else:
                    print(f"downloading {src} to {tar}")
                    os.system(f"hdfs dfs -get {src} {tar}")

    def freeze_params(self, freeze_prefix):
        for name, param in self.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    param.requires_grad = False

    def maxpooling_with_mask(self, hidden_state, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).half()
        mask_expanded = 1e4 * (mask_expanded - 1)
        hidden_masked = hidden_state + mask_expanded  # sum instead of multiple
        max_pooling = torch.max(hidden_masked, dim=1)[0]

        return max_pooling

    def multi_heads_with_mask(self, hidden_state, head_mask):
        x = self.multi_heads(hidden_state).reshape(-1, self.hparams.head_num, self.hparams.class_num)
        head_mask = head_mask.unsqueeze(-1).expand(x.size()).half()
        head_mask = 1e4 * (head_mask - 1)
        x = head_mask + x
        x = torch.max(x, dim=1)[0]
        return x

    def training_step(self, batch, idx):
        token_ids, token_mask, token_seg, token_pos, image, image_mask, label = (
            batch["input_ids"],
            batch["input_mask"],
            batch["input_seg"],
            batch["input_pos"],
            batch["image"],
            batch["image_mask"],
            batch["label"],
        )

        text_out = self.text(
            input_ids=token_ids,
            attention_mask=token_mask,
            token_type_ids=token_seg,
            position_ids=token_pos,
        )

        # text feature
        t_hidden = text_out["last_hidden_state"]  # text_len, feat_dim
        t_rep = t_hidden[:, 0, :]

        # image feature
        bz, n, c, h, w = image.shape
        v_hidden = self.visual(image.reshape(-1, c, h, w)).reshape(bz, n, -1)  # bz, n, feat_dim
        v_maxpool = self.maxpooling_with_mask(v_hidden, image_mask)

        # fuse feature
        vh2t = self.v_projector(v_hidden)  # map vhidden to thidden
        concat_tv = torch.cat([t_hidden, vh2t], dim=1)  # bz, length, hidden
        concat_mask = torch.cat([token_mask, image_mask], dim=1)
        pos_embed = self.pos_embedding(torch.arange(concat_tv.shape[1], device=concat_tv.device))
        all_hidden, prob = self.fuse(concat_tv + pos_embed, concat_mask)
        fuse_hidden = all_hidden[-1]
        f_rep = fuse_hidden[:, 0, :]

        # text cls
        t_logits = self.t_header(t_rep)
        loss_t = self.ce(t_logits, label)

        # image cls
        v_logits = self.v_header(v_maxpool)
        loss_v = self.ce(v_logits, label)

        # fuse cls
        f_logits = self.f_header(f_rep)
        # if self.hparams.use_multihead:
        #     head_mask = batch["head_mask"]
        #     f_logits = self.multi_heads_with_mask(f_logits, head_mask)
        # else:
        #     f_logits = self.f_cls(f_logits)
        loss_f = self.ce(f_logits, label)

        # loss and monitor
        loss = loss_t + loss_v + loss_f

        t_acc = self.cal_acc(t_logits, label=label, topk=(1,))
        # t_macc = self.cal_multilabel_acc(t_logits, multi_labels=batch["multi_labels"])
        v_acc = self.cal_acc(v_logits, label=label, topk=(1,))
        # v_macc = self.cal_multilabel_acc(v_logits, multi_labels=batch["multi_labels"])
        f_acc = self.cal_acc(f_logits, label=label, topk=(1,))
        # f_macc = self.cal_multilabel_acc(f_logits, multi_labels=batch["multi_labels"])

        res = {
            "loss": loss,
            "loss_t": loss_t,
            "loss_v": loss_v,
            "loss_f": loss_f,
            "train_t_top1": t_acc["top1_acc"],
            "train_v_top1": v_acc["top1_acc"],
            "train_f_top1": f_acc["top1_acc"],
            # 'train_t_macc': t_macc['multi_label_acc'],
            # 'train_v_macc': v_macc['multi_label_acc'],
            # 'train_f_macc': f_macc['multi_label_acc'],
            "train_lr": self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0],
        }

        return res

    def validation_step(self, batch, idx):
        token_ids, token_mask, token_seg, token_pos, image, image_mask, label = (
            batch["input_ids"],
            batch["input_mask"],
            batch["input_seg"],
            batch["input_pos"],
            batch["image"],
            batch["image_mask"],
            batch["label"],
        )

        text_out = self.text(
            input_ids=token_ids,
            attention_mask=token_mask,
            token_type_ids=token_seg,
            position_ids=token_pos,
        )

        # text feature
        t_hidden = text_out["last_hidden_state"]  # text_len, feat_dim
        t_rep = t_hidden[:, 0, :]

        # image feature
        bz, n, c, h, w = image.shape
        v_hidden = self.visual(image.reshape(-1, c, h, w)).reshape(bz, n, -1)  # bz, n, feat_dim
        v_maxpool = self.maxpooling_with_mask(v_hidden, image_mask)

        # fuse feature
        vh2t = self.v_projector(v_hidden)  # map vhidden to thidden
        concat_tv = torch.cat([t_hidden, vh2t], dim=1)  # bz, length, hidden
        concat_mask = torch.cat([token_mask, image_mask], dim=1)
        pos_embed = self.pos_embedding(torch.arange(concat_tv.shape[1], device=concat_tv.device))
        all_hidden, prob = self.fuse(concat_tv + pos_embed, concat_mask)
        fuse_hidden = all_hidden[-1]
        f_rep = fuse_hidden[:, 0, :]

        # text cls
        t_logits = self.t_header(t_rep)
        loss_t = self.ce(t_logits, label)

        # image cls
        v_logits = self.v_header(v_maxpool)
        loss_v = self.ce(v_logits, label)

        # fuse cls
        f_logits = self.f_header(f_rep)
        # if self.hparams.use_multihead:
        #     head_mask = batch["head_mask"]
        #     f_logits = self.multi_heads_with_mask(f_logits, head_mask)
        # else:
        #     f_logits = self.f_cls(f_logits)
        loss_f = self.ce(f_logits, label)

        # loss and monitor
        loss = loss_t + loss_v + loss_f

        t_acc = self.cal_acc(t_logits, label=label, topk=(1,))
        # t_macc = self.cal_multilabel_acc(t_logits, multi_labels=batch["multi_labels"])
        v_acc = self.cal_acc(v_logits, label=label, topk=(1,))
        # v_macc = self.cal_multilabel_acc(v_logits, multi_labels=batch["multi_labels"])
        f_acc = self.cal_acc(f_logits, label=label, topk=(1,))
        # f_macc = self.cal_multilabel_acc(f_logits, multi_labels=batch["multi_labels"])

        res = {
            "val_loss": loss,
            "val_loss_t": loss_t,
            "val_loss_v": loss_v,
            "val_loss_f": loss_f,
            "val_t_top1": t_acc["top1_acc"],
            "val_v_top1": v_acc["top1_acc"],
            "val_f_top1": f_acc["top1_acc"],
            # 'val_t_macc': t_macc['multi_label_acc'],
            # 'val_v_macc': v_macc['multi_label_acc'],
            # 'val_f_macc': f_macc['multi_label_acc'],
        }

        return res

    def infer_step(self, token_ids, token_mask, image, image_mask):
        token_pos = torch.tensor([list(range(token_ids.shape[-1]))], device=token_ids.device).repeat(
            token_ids.shape[0], 1
        )
        token_seg = torch.zeros_like(token_ids, device=token_ids.device)

        text_out = self.text(
            input_ids=token_ids,
            attention_mask=token_mask,
            token_type_ids=token_seg,
            position_ids=token_pos,
        )

        # text feature
        t_hidden = text_out["last_hidden_state"]  # text_len, feat_dim
        t_rep = t_hidden[:, 0, :]

        # image feature
        bz, n, c, h, w = image.shape
        v_hidden = self.visual(image.reshape(-1, c, h, w)).reshape(bz, n, -1)  # bz, n, feat_dim
        v_maxpool = self.maxpooling_with_mask(v_hidden, image_mask)

        # fuse feature
        vh2t = self.v_projector(v_hidden)  # map vhidden to thidden
        concat_tv = torch.cat([t_hidden, vh2t], dim=1)  # bz, length, hidden
        concat_mask = torch.cat([token_mask, image_mask], dim=1)
        pos_embed = self.pos_embedding(torch.arange(concat_tv.shape[1], device=concat_tv.device))
        all_hidden, prob = self.fuse(concat_tv + pos_embed, concat_mask)
        fuse_hidden = all_hidden[-1]
        f_rep = fuse_hidden[:, 0, :]

        softmax = nn.Softmax(dim=1)
        # rep to class
        t_logits = self.t_header(t_rep)
        t_score = softmax(t_logits)
        v_logits = self.v_header(v_maxpool)
        v_score = softmax(v_logits)
        f_logits = self.f_header(f_rep)
        f_score = softmax(f_logits)

        out = {
            "f_score": f_score,
            "t_score": t_score,
            "v_score": v_score,
            "f_rep": f_rep,
            "t_rep": t_rep,
            "v_rep": v_maxpool,
        }

        return out

    def trace_before_step(self, batch):
        # batch为dataloader的输出，一般为dict形式
        # 在trace_before_step中需要将dict形式的batch拆成list或tuple，再传入trace_step
        token_ids, token_mask, image, image_mask = (
            batch["input_ids"],
            batch["input_mask"],
            # batch["input_seg"],
            # batch["input_pos"],
            batch["image"],
            batch["image_mask"],
        )
        return token_ids, token_mask, image, image_mask

    def trace_step(self, batch):
        # batch为list或tuple
        # 在本方法中实现推理，并输出希望得到的推理结果，如logits
        token_ids, token_mask, image, image_mask = batch
        out = self.infer_step(token_ids=token_ids, token_mask=token_mask, image=image, image_mask=image_mask)
        return out["f_score"], out["t_score"], out["v_score"], out["f_rep"], out["t_rep"], out["v_rep"]

    def configure_optimizers(self):
        no_decay = ["bias", "bn", "norm"]
        no_dacay_params_dict = {"params": [], "weight_decay": 0.0}
        low_lr_params_dict = {
            "params": [],
            "weight_decay": self.hparams.weight_decay,
            "lr": self.hparams.learning_rate * 0.01,
        }
        normal_params_dict = {
            "params": [],
            "weight_decay": self.hparams.weight_decay,
        }

        low_lr_keys = []
        for n, p in self.named_parameters():
            low_lr = False
            for low_lr_prefix in self.hparams.low_lr_prefix:
                if n.startswith(low_lr_prefix):
                    low_lr = True
                    low_lr_params_dict["params"].append(p)
                    low_lr_keys.append(n)
                    break
            if low_lr:
                continue

            if any(nd in n for nd in no_decay):
                no_dacay_params_dict["params"].append(p)
            # elif n.startswith("albert"):
            #     low_lr_params_dict["params"].append(p)
            else:
                normal_params_dict["params"].append(p)

        if low_lr_keys:
            print(f"low_lr_keys are: {low_lr_keys}")

        optimizer_grouped_parameters = [
            no_dacay_params_dict,
            low_lr_params_dict,
            normal_params_dict,
        ]

        if self.hparams.optim == "SGD":
            optm = torch.optim.SGD(
                optimizer_grouped_parameters,
                self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optim == "AdamW":
            optm = AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-6,
                weight_decay=self.hparams.weight_decay,
                correct_bias=False,
            )

        if self.hparams.lr_schedule == "linear":
            print(f"warmup: {self.hparams.warmup_steps_factor * self.trainer.steps_per_epoch}")
            print(f"total step: {self.trainer.total_steps}")
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optm,
                num_warmup_steps=int(self.hparams.warmup_steps_factor * self.trainer.steps_per_epoch),
                num_training_steps=self.trainer.total_steps,
            )
        elif self.hparams.lr_schedule == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optm,
                num_warmup_steps=int(self.hparams.warmup_steps_factor * self.trainer.steps_per_epoch),
                num_training_steps=self.trainer.total_steps,
            )
        elif self.hparams.lr_schedule == "onecycle":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optm,
                max_lr=self.hparams.learning_rate,
                total_steps=self.trainer.total_steps,
            )

        return {"optimizer": optm, "lr_scheduler": lr_scheduler}

    def lr_scheduler_step(
        self,
        schedulers,
        **kwargs,
    ) -> None:
        """
        默认是per epoch的lr schedule, 改成per step的
        """
        for scheduler in schedulers:
            scheduler.step()

    @torch.no_grad()
    def cal_acc(self, output: torch.Tensor, label: torch.Tensor, topk=(1,)):
        """
        Computes the accuracy over the k top predictions for the specified values of k
        """
        maxk = max(topk)
        batch_size = label.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res[f"top{k}_acc"] = correct_k.mul_(100.0 / batch_size)
        return res

    @torch.no_grad()
    def cal_multilabel_acc(self, output: torch.Tensor, multi_labels: torch.Tensor()):
        _, pred = output.topk(1, 1, True, True)
        acc = torch.sum(pred.eq(multi_labels)) / pred.shape[0]
        res = {"multi_label_acc": float(100.0 * acc)}
        return res
