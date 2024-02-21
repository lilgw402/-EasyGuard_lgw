# -*- coding: utf-8 -*-
try:
    import cruise
except ImportError:
    print(
        "[ERROR] cruise is not installed! Please refer this doc: https://bytedance.feishu.cn/wiki/wikcnGP7yzZAuKpPfL6jRJKl2ag"
    )
import torch.nn.functional as F
from cruise import CruiseModule
from cruise.utilities.distributed import DIST_ENV
from torch import nn

# import soundfile
from tqdm import tqdm

from easyguard.core.optimizers import *
from easyguard.core.optimizers import AdamW

from .ecapatdnn import ECAPA_TDNN
from .loss import AAMsoftmax, LearnableNTXentLoss
from .tools import *


class FashionSV(CruiseModule):
    def __init__(
        self,
        mode: str = "aam",
        class_num: int = 2100,
        hidden_dim: int = 192,
        channel: int = 512,
        m: float = 0.2,
        s: float = 30,
        optim: str = "AdamW",
        learning_rate: float = 1.0e-4,
        weight_decay: float = 1.0e-4,
        lr_schedule: str = "linear",
        warmup_steps_factor: float = 4,
        low_lr_prefix: list = [],
        freeze_prefix: list = [],
        load_pretrained: str = None,
        prefix_changes: list = [],
        download_files: list = [],
        **kwargs,
    ):
        super(FashionSV, self).__init__()
        self.save_hparams()

    def setup(self, stage):
        # ECAPA-TDNN
        self.speaker_encoder = ECAPA_TDNN(C=self.hparams.channel, hidden_dim=self.hparams.hidden_dim)
        # Classifier
        mode = self.hparams.mode
        if mode == "aam":
            self.speaker_loss = AAMsoftmax(
                n_class=self.hparams.class_num,
                m=self.hparams.m,
                s=self.hparams.s,
                hidden_dim=self.hparams.hidden_dim,
            )
        elif mode == "cls":
            self.classifier = nn.Linear(self.hparams.hidden_dim, self.hparams.class_num)
            self.ce = nn.CrossEntropyLoss()
        elif mode == "clip":
            self.cl_loss = LearnableNTXentLoss()
        else:
            raise Exception(f"unknown training mode, only support aam, cls and clip for now")

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
        if self.hparams.download_files:
            import os

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

    def forward_step(self, feature):
        speaker_embedding = self.speaker_encoder.forward(feature, aug=True)  # audio feature
        return speaker_embedding

    def training_step(self, batch, idx):
        feature = batch["feature"]
        labels = batch["labels"]
        if self.hparams.mode == "aam":
            speaker_embedding = self.forward_step(feature)
            # allgather
            speaker_embedding = self.all_gather(speaker_embedding.contiguous())
            speaker_embedding = speaker_embedding.flatten(0, 1)
            labels = self.all_gather(labels.contiguous())
            labels = labels.flatten(0, 1)
            #
            nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)
            rep_dict = {
                "loss": nloss,
                "prec": prec,
                "train_lr": self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0],
            }
        elif self.hparams.mode == "cls":
            speaker_embedding = self.forward_step(feature)
            logits = self.classifier(speaker_embedding)
            loss = self.ce(logits, labels)
            rep_dict = {
                "loss": loss,
                "train_lr": self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0],
            }
        elif self.hparams.mode == "clip":
            splitlen = feature.shape[1] // 2
            sf1, sf2 = feature[:, :splitlen], feature[:, splitlen:]
            se1 = self.forward_step(sf1)
            se2 = self.forward_step(sf2)
            allgather_se1 = self.all_gather(se1.contiguous())
            allgather_se1 = allgather_se1.flatten(0, 1)
            allgather_se2 = self.all_gather(se2.contiguous())
            allgather_se2 = allgather_se2.flatten(0, 1)
            cl_loss = self.cl_loss(allgather_se1, allgather_se2)
            rep_dict = {
                "loss": cl_loss,
                "train_lr": self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0],
            }
        else:
            raise Exception(f"only support aam and cls")

        return rep_dict

    def validation_step(self, batch, idx):
        feature = batch["feature"]
        labels = batch["labels"]
        if self.hparams.mode == "aam":
            speaker_embedding = self.forward_step(feature)
            # allgather
            speaker_embedding = self.all_gather(speaker_embedding.contiguous())
            speaker_embedding = speaker_embedding.flatten(0, 1)
            labels = self.all_gather(labels.contiguous())
            labels = labels.flatten(0, 1)
            #
            nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)
            rep_dict = {
                "val_loss": nloss,
                "val_prec": prec,
            }
        elif self.hparams.mode == "cls":
            speaker_embedding = self.forward_step(feature)
            logits = self.classifier(speaker_embedding)
            loss = self.ce(logits, labels)
            rep_dict = {
                "val_loss": loss,
            }
        elif self.hparams.mode == "clip":
            splitlen = feature.shape[1] // 2
            sf1, sf2 = feature[:, :splitlen], feature[:, splitlen:]
            se1 = self.forward_step(sf1)
            se2 = self.forward_step(sf2)
            allgather_se1 = self.all_gather(se1.contiguous())
            allgather_se1 = allgather_se1.flatten(0, 1)
            allgather_se2 = self.all_gather(se2.contiguous())
            allgather_se2 = allgather_se2.flatten(0, 1)
            cl_loss = self.cl_loss(allgather_se1, allgather_se2)
            rep_dict = {
                "val_loss": cl_loss,
            }
        else:
            raise Exception(f"only support aam and cls")

        return rep_dict

    def validation_epoch_end(self, outputs):
        gathered_results = DIST_ENV.all_gather_object(outputs)

        res_out = dict()
        all_results = []
        for item in gathered_results:
            all_results.extend(item)
        val_loss_all = [out["val_loss"] for out in all_results]
        # val_prec_all = [out["val_prec"] for out in all_results]

        val_loss = sum(val_loss_all) / len(val_loss_all)
        # val_prec = sum(val_prec_all) / len(val_prec_all)

        res_out["val_loss"] = val_loss
        # res_out["val_prec"] = val_prec

        self.log_dict(res_out, console=True)
        self.log("val_loss", val_loss, console=True)
        # self.log("val_prec", val_prec, console=True)

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

    # def eval_network(self, eval_list, eval_path):
    #     self.eval()
    #     files = []
    #     embeddings = {}
    #     lines = open(eval_list).read().splitlines()
    #     for line in lines:
    #         files.append(line.split()[1])
    #         files.append(line.split()[2])
    #     setfiles = list(set(files))
    #     setfiles.sort()
    #
    #     for idx, file in tqdm(enumerate(setfiles), total=len(setfiles)):
    #         audio, _ = soundfile.read(os.path.join(eval_path, file))
    #         # Full utterance
    #         data_1 = torch.FloatTensor(np.stack([audio], axis=0)).cuda()
    #
    #         # Spliited utterance matrix
    #         max_audio = 300 * 160 + 240
    #         if audio.shape[0] <= max_audio:
    #             shortage = max_audio - audio.shape[0]
    #             audio = np.pad(audio, (0, shortage), 'wrap')
    #         feats = []
    #         startframe = np.linspace(0, audio.shape[0] - max_audio, num=5)
    #         for asf in startframe:
    #             feats.append(audio[int(asf):int(asf) + max_audio])
    #         feats = np.stack(feats, axis=0).astype(np.float)
    #         data_2 = torch.FloatTensor(feats).cuda()
    #         # Speaker embeddings
    #         with torch.no_grad():
    #             embedding_1 = self.speaker_encoder.forward(data_1, aug=False)
    #             embedding_1 = F.normalize(embedding_1, p=2, dim=1)
    #             embedding_2 = self.speaker_encoder.forward(data_2, aug=False)
    #             embedding_2 = F.normalize(embedding_2, p=2, dim=1)
    #         embeddings[file] = [embedding_1, embedding_2]
    #     scores, labels = [], []
    #
    #     for line in lines:
    #         embedding_11, embedding_12 = embeddings[line.split()[1]]
    #         embedding_21, embedding_22 = embeddings[line.split()[2]]
    #         # Compute the scores
    #         score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
    #         score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
    #         score = (score_1 + score_2) / 2
    #         score = score.detach().cpu().numpy()
    #         scores.append(score)
    #         labels.append(int(line.split()[0]))
    #
    #     # Coumpute EER and minDCF
    #     EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
    #     fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    #     minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
    #
    #     return EER, minDCF
