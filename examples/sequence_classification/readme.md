# Sequence classification examples with FashionXLM and RoBERTa

## Launch gpu worker:

```shell
launch --gpu 1 --cpu 32 --memory 64 -- doas --krb5-username [EMAIL PREFIX] bash
```
## Finetune:

``` shell
cd EasyGuard/examples/sequence_classification/

# FashionXLM
python3 run_model.py --config config/config_fashionxlm_base.yaml

# FashionXLM-MOE
python3 run_model.py --config config/config_fashionxlm_moe_base.yaml

# XLM-RoBERTa
python3 run_model.py --config config/config_xlmr_base.yaml
```

## Results[ASNA]

| Model          | f1 score  |
| -------------- | --------- |
| FashionXLM     | 0.928125  |
| FashionXLM-MOE | 0.9296875 |
| XLM-RoBERTa    | 0.875     |

## Feature Extract:

```
cd Easyguard

python3 ./examples/sequence_classification/run_feature_extractor.py
```
