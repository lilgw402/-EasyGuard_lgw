# -*- coding: utf-8 -*-
from typing import Optional

import torch
import torch.nn as nn
from ptx.model import Model


class DebertaClassifier(Model):
    def __init__(self, config, num_labels: int = 2):
        super().__init__()
        self.deberta = Model.from_option(f"file:{config}|strict=false")

        self.dropout = torch.nn.Dropout(0.1)
        self.num_labels = num_labels
        self.classifier = torch.nn.Linear(in_features=self.deberta.encoder.dim, out_features=self.num_labels)

        self.loss = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label: Optional[torch.Tensor] = None,
    ):
        pooled_output = self.deberta(
            input_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            output_pooled=True,
        )["pooled_output"]
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        if label is not None:
            loss = self.loss(logits.view(-1, self.num_labels), label.view(-1))
            return logits, loss
        else:
            softmax = torch.nn.Softmax(dim=1)
            # softmax_score:        [batch_size, num_classes]
            softmax_score = softmax(logits)
            return softmax_score.cpu()
