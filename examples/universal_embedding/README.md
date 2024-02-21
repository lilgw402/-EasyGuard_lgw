# This is an example of fashion-universal demo.

https://bytedance.feishu.cn/docx/TjxVd6ne9o5uAMxocScclwpxnOi

### Model Selection
```
[fashion-universal-vit-base-224]: universal embedding
[fashion-universal-product-vit-base-224]: product universal embedding
```

### Feature extraction by auto model:
```
python3 automodel/feature_extractor.py
```

### Distributed training:
```
sh scripts/universal_vit_b_16.sh
```