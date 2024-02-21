# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-02-27 20:25:14
# Modified: 2023-02-27 20:25:14
import json

import torch.nn as nn
from titan import create_model


class EcomDebertaEncoder(nn.Module):
    def __init__(self, config_file, **kwargs):
        super().__init__()
        self.config = self.init_config(config_file)
        self.config.update(kwargs)
        self.mode = self.config.get("mode", "cls")
        self.source = self.config.get("source", "titan")
        self.deberta = create_model(
            model_name=self.model_name,
            pretrained=True,
            pretrained_uri=self.pretrained_model_path,
            n_layers=self.config["num_hidden_layers"],
        )
        # classifier_dropout = (0 if 'hidden_dropout_prob' not in
        # self.config else self.config.get('hidden_dropout_prob',0))
        # self.dropout = nn.Dropout(classifier_dropout)
        # self.deberta.eval()

    def init_config(self, config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
        self.model_name = config.get("model_name", "deberta_base_6l")
        self.pretrained_model_path = config.get(
            "pretrained_model_path",
            "/opt/tiger/models/weights/fashion_deberta_word/epoch9.ckpt",
        )
        return config

    def forward(self, input_ids, input_masks, input_segment_ids):
        """
        search-deberta usage
        USE output['pooled_output'] OR output['sequence_output']
        eg:
        pooled_output = self.deberta(input_ids, attention_mask, segment_ids, pinyin_ids)['pooled_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        """

        if self.source == "titan":
            # with torch.no_grad():
            output = self.deberta(
                input_ids=input_ids,
                attention_mask=input_masks,
                segment_ids=input_segment_ids,
                output_pooled=True,
            )
            if self.mode == "pool":
                return output["pooled_output"]
            elif self.mode == "cls":
                return output["sequence_output"][:, 0, :]
            else:
                raise ("Mode {} not found error".format(self.mode))
        elif self.source == "hf":
            # huggingface usage
            # with torch.no_grad():
            output = self.deberta(input_ids, input_masks, input_segment_ids)
            if self.mode == "cls":
                last_hidden_states = output.hidden_states[-1]
                return last_hidden_states
            else:
                return output.hidden_states[0]
        else:
            raise ("Source {} not found error".format(self.source))
