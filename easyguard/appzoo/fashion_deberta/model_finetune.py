import numpy as np
import torch
import torch.nn as nn
from cruise import CruiseModule
from cruise.utilities.distributed import DIST_ENV
from sklearn import metrics

from easyguard.core import AutoModel

from ...utils.losses import SCELoss


class FashionDebertaFtModel(CruiseModule):
    def __init__(
        self,
        learning_rate: float = 2e-5,
        pretrain_model_name="fashion-deberta-asr",
        cls_class_num: int = 2,
        sce_loss_enable: bool = False,
        auc_score_enable: bool = True,  # need to guarantee eval dataset contains all class label
        hidden_size: int = 768,
    ):
        super().__init__()
        self.save_hparams()

    def setup(self, stage) -> None:
        self.deberta = AutoModel.from_pretrained(self.hparams.pretrain_model_name, rm_deberta_prefix=True)

        self.cls_prediction = nn.Linear(self.hparams.hidden_size, self.hparams.cls_class_num)

        self.cls_loss_fct = (
            SCELoss(alpha=1.0, beta=0.5, num_classes=self.hparams.cls_class_num)
            if self.hparams.sce_loss_enable
            else nn.CrossEntropyLoss(ignore_index=-100)
        )

        self.train_return_loss_dict = dict()
        self.val_return_loss_dict = dict()

        # self.partial_load_from_checkpoints(os.path.join(self.local_pretrained_model_dir, "model.ckpt"))

    def training_step(self, batch, idx):
        try:
            (
                input_ids,
                input_masks,
                input_segment_ids,
                classification_labels,
            ) = (
                batch["input_ids"],
                batch["input_masks"],
                batch["input_segment_ids"],
                batch["classification_labels"],
            )
            output = self.deberta(
                input_ids=input_ids,
                segment_ids=input_segment_ids,
                attention_mask=input_masks,
                output_pooled=True,
            )
            classification_pred_logits = self.cls_prediction(output["pooled_output"])
            # cls loss
            cls_loss = self.cls_loss_fct(
                classification_pred_logits.reshape(-1, self.hparams.cls_class_num),
                classification_labels.reshape(-1),
            )
            self.train_return_loss_dict["loss"] = cls_loss
        except Exception as e:
            print(f"train error: {str(e)}")
        return self.train_return_loss_dict

    def validation_step(self, batch, batch_idx):
        input_ids, input_masks, input_segment_ids, classification_labels = (
            batch["input_ids"],
            batch["input_masks"],
            batch["input_segment_ids"],
            batch["classification_labels"],
        )
        output = self.deberta(
            input_ids=input_ids,
            segment_ids=input_segment_ids,
            attention_mask=input_masks,
            output_pooled=True,
        )
        classification_pred_logits = self.cls_prediction(output["pooled_output"])
        # cls loss
        cls_loss = self.cls_loss_fct(
            classification_pred_logits.reshape(-1, self.hparams.cls_class_num),
            classification_labels.reshape(-1),
        )
        self.val_return_loss_dict["val_loss"] = cls_loss
        # prepare for auc
        if self.hparams.auc_score_enable:
            # multi class y_scores require (n_sample, n_class) shape
            if self.hparams.cls_class_num > 2:
                pred_probs = nn.Softmax(-1)(classification_pred_logits)
            # 2 class y_scores require (n_sample,) shape
            else:
                pred_probs = nn.Softmax(-1)(classification_pred_logits)[:, -1]
            self.val_return_loss_dict["pred_probs"] = pred_probs.cpu().detach().numpy()
            self.val_return_loss_dict["classification_labels"] = classification_labels.cpu().detach().numpy()
        return self.val_return_loss_dict

    def validation_epoch_end(self, outputs) -> None:
        gathered_results = DIST_ENV.all_gather_object(outputs)
        all_results = []
        for item in gathered_results:
            all_results.extend(item)
        # auc score
        if self.hparams.auc_score_enable:
            classification_labels_all = np.concatenate([out["classification_labels"] for out in all_results], axis=0)
            pred_probs_all = np.concatenate([out["pred_probs"] for out in all_results], axis=0)
            # multi class
            if self.hparams.cls_class_num > 2:
                auc_score = metrics.roc_auc_score(
                    classification_labels_all,
                    pred_probs_all,
                    multi_class="ovr",
                )
            else:
                auc_score = metrics.roc_auc_score(classification_labels_all, pred_probs_all)
            self.log("total_auc_score", auc_score, console=True)
            print("total_auc_score", auc_score)

    def forward(self, input_ids, input_masks, input_segment_ids):
        output = self.deberta(
            input_ids=input_ids,
            segment_ids=input_segment_ids,
            attention_mask=input_masks,
            output_pooled=True,
        )
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return {"optimizer": optimizer}

    def trace_before_step(self, batch):
        # tracer don't like dict of input
        x = [
            batch["input_ids"],
            batch["input_segment_ids"],
            batch["input_masks"],
        ]
        return x

    def trace_step(self, input_ids, input_segment_ids, input_masks):
        return self(input_ids, input_segment_ids, input_masks)
