# This is an example of FashionVTP pretrain and finetune demo.

```python
# modify the config file
vi examples/fashion_vtp/un_hq_configs/auto_default_config.yaml
# if no config file found, can dump default configs as initial file (in your local machine)
python3 examples/fashion_vtp/run_fashionvtp_with_fashionbert_finetuning.py  --print_config > examples/fashion_vtp/un_hq_configs/auto_default_config.yaml
```
## Finetune

### Model build:
```
from easyguard.core import AutoModel
class MyModel(CruiseModule):
    def __init__(
        self,
        ...
    ):
        super(MyModel, self).__init__()
        self.save_hparams()

    def setup(self, stage) -> None:
        ...
        """
            FashionVTP includes: 
                "fashionvtp-base", 
                "fashionvtp-base-c",
                "fashionvtp-base-s"
        """
        self.fashionvtp_model = AutoModel.from_pretrained("fashionvtp-base")
        self.fashionbert_model = AutoModel.from_pretrained("fashionbert-base")

        # fashionbert:512, fashionvtp:768
        self.reduc = nn.Linear(768+512, 768) 
        self.classifier = torch.nn.Linear(768, self.config_optim.class_num)
```

### Training:
```
/path/to/EasyGuard/tools/TORCHRUN examples/fashion_vtp/run_fashionvtp_with_fashionbert_finetuning.py \
    --config examples/fashion_vtp/un_hq_configs/auto_default_config.yaml
```

## Pretrain

Only support distributed training

```
/path/to/EasyGuard/tools/TORCHRUN ./examples/fashion_vtp/run_fashionvtp_pretrain.py \
    --config ./examples/fashion_vtp/pretrain_configs/default_config.yaml
```

## Doc
[**电商预训练--视频多模态FashionVTPv3**](https://bytedance.feishu.cn/docx/GQlRd5J65oCOQBxRLqcc3qXBnth)