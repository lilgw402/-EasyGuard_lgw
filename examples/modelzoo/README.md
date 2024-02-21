Pre-trained models

* run EasyGuard model

``` python
    from easyguard.core import AutoModel, AutoTokenizer
    my_model = AutoModel.from_pretrained("fashion-deberta")
```

* run titan model

``` python
    model = create_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        pretrained_version=args.model_version,
        features_only=args.features_only,
        tos_helper=tos_helper,
    )
```

