# -*- coding: utf-8 -*-
"""
FrameAlbert Classification
"""
try:
    import cruise
except ImportError:
    print(
        "[ERROR] cruise is not installed! Please refer this doc: https://bytedance.feishu.cn/wiki/wikcnGP7yzZAuKpPfL6jRJKl2ag"
    )
from cruise import CruiseModule

# from cruise.utilities.cloud_io import load
from cruise.utilities.distributed import DIST_ENV

from easyguard import AutoModel

# from easyguard.modelzoo.models.falbert import FalBertModel as FrameALBert
from easyguard.core.optimizers import *
from easyguard.core.optimizers import AdamW


class FrameAlbertClassify(CruiseModule):
    def __init__(
        self,
        backbone="fashionproduct-xl-general-v1",
        class_num: int = 2100,
        hidden_dim: int = 768,
        optim: str = "AdamW",
        learning_rate: float = 1.0e-4,
        weight_decay: float = 1.0e-4,
        lr_schedule: str = "linear",
        warmup_steps_factor: float = 4,
        low_lr_prefix: list = [],
        freeze_prefix: list = [],
        head_num: int = 5,
        use_multihead: bool = True,
        load_pretrained: str = None,
        prefix_changes: list = [],
        download_files: list = [],
    ):
        super(FrameAlbertClassify, self).__init__()
        self.save_hparams()

    def setup(self, stage) -> None:
        """
        Initialize modules
        """
        # self.falbert = FrameALBert(self.config_backbone)
        self.backbone = AutoModel.from_pretrained(self.hparams.backbone)
        """
        Initialize output layer
        """
        hidden_dim = self.hparams.hidden_dim
        if self.hparams.use_multihead:
            self.multi_heads = torch.nn.Linear(hidden_dim * 2, self.hparams.head_num * self.hparams.class_num)
        else:
            self.classifier_concat = torch.nn.Linear(hidden_dim * 2, self.hparams.class_num)

        self.ce = torch.nn.CrossEntropyLoss()
        """
        Initialize some fixed parameters.
        """

        if self.hparams.load_pretrained:
            prefix_changes = [prefix_change.split("->") for prefix_change in self.hparams.prefix_changes]
            rename_params = {pretrain_prefix: new_prefix for pretrain_prefix, new_prefix in prefix_changes}
            self.partial_load_from_checkpoints(
                self.hparams.load_pretrained,
                map_location="cpu",
                rename_params=rename_params,
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

    # def init_weights(self):
    #     def init_weight_module(module):
    #         if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
    #             torch.nn.init.xavier_uniform_(module.weight)
    #         elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
    #             module.bias.data.zero_()
    #             module.weight.data.fill_(1.0)
    #         if isinstance(module, torch.nn.Linear) and module.bias is not None:
    #             module.bias.data.zero_()
    #
    #     self.apply(init_weight_module)

    def freeze_params(self, freeze_prefix):
        for name, param in self.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    param.requires_grad = False

    def criterion(self, logits, label):
        loss = self.ce(logits, label)
        return loss

    # def maxpooling_with_mask(self, hidden_state, attention_mask):
    #     mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).half()
    #     mask_expanded = 1e4 * (mask_expanded - 1)
    #     hidden_masked = hidden_state + mask_expanded  # sum instead of multiple
    #     max_pooling = torch.max(hidden_masked, dim=1)[0]
    #
    #     return max_pooling

    # def meanpooling_with_mask(self, hidden_state, attention_mask):
    #     masked_last_hidden = hidden_state * attention_mask.unsqueeze(-1)
    #     sum_last_hidden = torch.sum(masked_last_hidden, dim=1)
    #     sum_mask = torch.sum(attention_mask, dim=1).unsqueeze(-1)
    #     sum_mask = torch.clamp(sum_mask, min=1e-7).half()
    #     mean_pooling = sum_last_hidden / sum_mask
    #
    #     return mean_pooling

    def multi_heads_with_mask(self, hidden_state, head_mask):
        x = self.multi_heads(hidden_state).reshape(-1, self.hparams.head_num, self.hparams.class_num)
        head_mask = head_mask.unsqueeze(-1).expand(x.size()).half()
        head_mask = 1e4 * (head_mask - 1)
        x = head_mask + x
        x = torch.max(x, dim=1)[0]
        return x

    def forward_step(
        self,
        input_ids,
        input_segment_ids,
        input_mask,
        frames=None,
        frames_mask=None,
        head_mask=None,
    ):
        rep_dict = self.backbone(
            input_ids=input_ids,
            input_segment_ids=input_segment_ids,
            input_mask=input_mask,
            frames=frames,
            frames_mask=frames_mask,
            output_hidden=False,
        )
        cls_emb = rep_dict["pooler"]
        max_pooling = rep_dict["max_pooling"]

        concat_feat = torch.cat([cls_emb, max_pooling], dim=1)

        if self.hparams.use_multihead:
            logits = self.multi_heads_with_mask(concat_feat, head_mask)
        else:
            logits = self.classifier_concat(concat_feat)

        return {"feat": concat_feat, "logits": logits}

    def training_step(self, batch, idx):
        token_ids, segment_ids, attn_mask, image, image_mask, head_mask = (
            batch["input_ids"],
            batch["input_segment_ids"],
            batch["input_mask"],
            batch["frames"],
            batch["frames_mask"],
            batch["head_mask"],
        )
        rep_dict = self.forward_step(
            input_ids=token_ids,
            input_segment_ids=segment_ids,
            input_mask=attn_mask,
            frames=image,
            frames_mask=image_mask,
            head_mask=head_mask,
        )
        rep_dict.update({"label": batch["label"]})
        loss = self.criterion(rep_dict["logits"], rep_dict["label"])
        res = {
            "loss": loss,
            "train_lr": self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0],
        }
        acc_dict = self.cal_acc(rep_dict["logits"], label=rep_dict["label"], topk=(1, 5))
        for k, v in acc_dict.items():
            res.update({f"train_{k}": v})

        return res

    def validation_step(self, batch, idx):
        token_ids, segment_ids, attn_mask, image, image_mask, head_mask = (
            batch["input_ids"],
            batch["input_segment_ids"],
            batch["input_mask"],
            batch["frames"],
            batch["frames_mask"],
            batch["head_mask"],
        )
        rep_dict = self.forward_step(
            input_ids=token_ids,
            input_segment_ids=segment_ids,
            input_mask=attn_mask,
            frames=image,
            frames_mask=image_mask,
            head_mask=head_mask,
        )
        rep_dict.update({"label": batch["label"]})
        loss = self.criterion(rep_dict["logits"], rep_dict["label"])
        res = {"val_loss": loss}

        acc_dict = self.cal_acc(rep_dict["logits"], label=rep_dict["label"], topk=(1, 5))
        for k, v in acc_dict.items():
            res.update({f"val_{k}": v})

        return res

    def validation_epoch_end(self, outputs) -> None:
        gathered_results = DIST_ENV.all_gather_object(outputs)

        res_out = {}
        all_results = []
        for item in gathered_results:
            all_results.extend(item)
        val_loss_all = [out["val_loss"] for out in all_results]
        top1_acc_all = [out["val_top1_acc"] for out in all_results]
        top5_acc_all = [out["val_top5_acc"] for out in all_results]

        val_loss = sum(val_loss_all) / len(val_loss_all)
        top1_acc = sum(top1_acc_all) / len(top1_acc_all)
        top5_acc = sum(top5_acc_all) / len(top5_acc_all)

        res_out["val_loss"] = val_loss
        res_out["val_top1_acc"] = top1_acc
        res_out["val_top5_acc"] = top5_acc

        self.log_dict(res_out, console=True)
        self.log("val_loss", val_loss, console=True)
        self.log("val_top1_acc", top1_acc, console=True)
        self.log("val_top5_acc", top1_acc, console=True)

    def trace_before_step(self, batch):
        # batch为dataloader的输出，一般为dict形式
        # 在trace_before_step中需要将dict形式的batch拆成list或tuple，再传入trace_step
        token_ids, segment_ids, attn_mask, image, image_mask, head_mask = (
            batch["input_ids"],
            batch["input_segment_ids"],
            batch["input_mask"],
            batch["frames"],
            batch["frames_mask"],
            batch["head_mask"],
        )
        return token_ids, segment_ids, attn_mask, image, image_mask, head_mask

    def trace_step(self, batch):
        # batch为list或tuple
        # 在本方法中实现推理，并输出希望得到的推理结果，如logits
        token_ids, segment_ids, attn_mask, image, image_mask, head_mask = batch
        rep_dict = self.forward_step(
            input_ids=token_ids,
            input_segment_ids=segment_ids,
            input_mask=attn_mask,
            frames=image,
            frames_mask=image_mask,
            head_mask=head_mask,
        )
        logits = rep_dict["logits"]
        return logits

    def trace_after_step(self, result):
        # 按照本文档导出无需实现该方法，留空即可
        pass

    def configure_optimizers(self):
        no_decay = ["bias", "bn", "norm"]
        no_dacay_params_dict = {"params": [], "weight_decay": 0.0}
        low_lr_params_dict = {
            "params": [],
            "weight_decay": self.hparams.weight_decay,
            "lr": self.hparams.learning_rate * 0.1,
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
    def cal_acc(
        self,
        output: torch.Tensor,
        label: torch.Tensor,
        topk: Tuple[int] = (1,),
    ):
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
