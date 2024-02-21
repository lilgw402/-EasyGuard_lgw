r"""
Build standard BERT style of models from hugging face.
"""
import torch
import torch.nn as nn
from transformers import BertModel


class RoBerta(nn.Module):
    __backbone_type__ = "bert"

    def __init__(
        self,
        transformer,
        bert_dir,
        mlm_enable,
        embedder_only=False,
        with_hidden_states=False,
        out_channels=768,
    ):
        r"""
        Args:
            transformer: the original bert style transformer from hugging face, like BERT, Roberta, etc.
            config: path to bert config on hdfs
            mlm_enable: if True, use Bert MLM for training
            out_channels: last out_channels of bert
            embedder_only: if True, only use the embedding step and return the feature together with the attn_mask
            with_hidden_states: if True, return the hidden states together with the features
        """
        super(RoBerta, self).__init__()
        print(f"=> Model arch: using {transformer.__name__} with config from {bert_dir}.")
        print(
            f"   Bert prams: mlm_enable={mlm_enable}; embedder_only={embedder_only}; "
            f"with_hidden_states={with_hidden_states}; out_channels={out_channels}"
        )

        self.model = transformer.from_pretrained(bert_dir)
        self.config = self.model.config  # record BertConfig for outer interface alignment

        self.embedder_only = embedder_only
        self.with_hidden_states = with_hidden_states
        self.last_out_channels = out_channels
        self.mlm_enable = mlm_enable

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        if isinstance(bert_output, tuple):  # compatibility with transformers < 4.0
            assert len(bert_output) == 2, bert_output
            hidden_states, features = bert_output
        else:
            hidden_states = bert_output.last_hidden_state
            features = bert_output.pooler_output

        if self.with_hidden_states:
            return hidden_states, features
        else:
            return features


if __name__ == "__main__":
    text_conifg = {
        "transformer": BertModel,
        "bert_dir": "./chinese_roberta_wwm_ext_pytorch",
        "mlm_enable": False,
        "embedder_only": False,
        "with_hidden_states": False,
        "out_channels": 768,
    }
    text_model = RoBerta(**text_conifg)
    print(text_model)
    input_ids = torch.LongTensor([[1] * 512])
    attention_mask = torch.LongTensor([[1] * 512])
    token_type_ids = torch.LongTensor([[0] * 512])
    print(text_model(input_ids, attention_mask, token_type_ids))
