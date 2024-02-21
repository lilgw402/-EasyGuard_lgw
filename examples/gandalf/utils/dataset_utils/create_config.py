from addict import Dict


def create_cruise_process_config(cfg, mode, is_kv=False):
    process_cfg_dict = {
        "custom_modals": "lightning_modal",
        "custom_op_modules": {
            "lightning_modal": "dataset.custom_dataset",
        },
        "modal_keys": {
            "lightning_modal": "all",
        },
        "custom_transforms": {
            "lightning_modal": {
                "transform": [],
                "batch_transform": [],
                "skip_collate": True,
            }
        },
    }
    if is_kv:
        process_cfg_dict["custom_transforms"]["lightning_modal"]["transform"].append(
            {"CruiseKVFeatureProvider": Dict(cfg.feature_provider)}
        )
        process_cfg_dict["custom_transforms"]["lightning_modal"]["batch_transform"].append(
            {"CruiseKVBatchFeatureProvider": Dict(cfg.feature_provider)}
        )
        if mode == "train":
            process_cfg_dict["custom_transforms"]["lightning_modal"]["transform"][0]["CruiseKVFeatureProvider"][
                "save_extra"
            ] = False
            process_cfg_dict["custom_transforms"]["lightning_modal"]["batch_transform"][0][
                "CruiseKVBatchFeatureProvider"
            ]["save_extra"] = False
        else:
            process_cfg_dict["custom_transforms"]["lightning_modal"]["transform"][0]["CruiseKVFeatureProvider"][
                "save_extra"
            ] = True
            process_cfg_dict["custom_transforms"]["lightning_modal"]["transform"][0]["CruiseKVFeatureProvider"][
                "eval_mode"
            ] = True
            process_cfg_dict["custom_transforms"]["lightning_modal"]["batch_transform"][0][
                "CruiseKVBatchFeatureProvider"
            ]["save_extra"] = True
            process_cfg_dict["custom_transforms"]["lightning_modal"]["batch_transform"][0][
                "CruiseKVBatchFeatureProvider"
            ]["eval_mode"] = True

    else:
        process_cfg_dict["custom_transforms"]["lightning_modal"]["transform"].append({"CruiseFakeProcessor": {}})
        process_cfg_dict["custom_transforms"]["lightning_modal"]["batch_transform"].append(
            {"CruiseFeatureProvider": Dict(cfg.feature_provider)}
        )
        if mode == "train":
            process_cfg_dict["custom_transforms"]["lightning_modal"]["batch_transform"][0]["CruiseFeatureProvider"][
                "save_extra"
            ] = False
        else:
            process_cfg_dict["custom_transforms"]["lightning_modal"]["batch_transform"][0]["CruiseFeatureProvider"][
                "save_extra"
            ] = True
            process_cfg_dict["custom_transforms"]["lightning_modal"]["batch_transform"][0]["CruiseFeatureProvider"][
                "eval_mode"
            ] = True

    return Dict(process_cfg_dict)
