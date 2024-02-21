# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-02-27 20:27:16
# Modified: 2023-02-27 20:27:16
from utils.registry import FEATURE_PROVIDERS
from utils.torch_util import collate_fields, default_collate

from .ecom_live_gandalf.ecom_live_gandalf_cruise_data_module import (
    EcomLiveGandalfParquetAutoDisCruiseDataModule,
    EcomLiveGandalfParquetAutoDisFeatureProvider,
)


def build_feature_provider(fp_type, arg_dict, save_extra=False, eval_mode=False, trace_mode=False):
    if isinstance(arg_dict, list):
        feature_provider = []
        for _arg_dict in arg_dict:
            _arg_dict.save_extra = save_extra
            _arg_dict.eval_mode = eval_mode
            _fp_type = _arg_dict.pop("type")
            constructor = FEATURE_PROVIDERS.get(_fp_type)
            try:
                _feature_provider = constructor(**_arg_dict, trace_mode=trace_mode)
            except TypeError:
                _feature_provider = constructor(**_arg_dict)
            feature_provider.append(_feature_provider)
    else:
        arg_dict.save_extra = save_extra
        arg_dict.eval_mode = eval_mode
        constructor = FEATURE_PROVIDERS.get(fp_type)
        try:
            feature_provider = constructor(**arg_dict, trace_mode=trace_mode)
        except TypeError:
            feature_provider = constructor(**arg_dict)
    return feature_provider


class CruiseFeatureProvider:
    def __init__(self, **fp_conf):
        from addict import Dict

        fp_conf = Dict(fp_conf)
        fp_type = fp_conf.pop("type")
        save_extra = fp_conf.get("save_extra", False)
        eval_mode = fp_conf.get("eval_mode", False)
        trace_mode = fp_conf.get("trace_mode", False)
        self.fp = build_feature_provider(fp_type, fp_conf, save_extra, eval_mode, trace_mode)
        if hasattr(self.fp, "collate"):
            self.collate_fn = self.fp.collate
        else:
            self.collate_fn = collate_fields

    def __call__(self, batch_data):
        batch_processed_data = self.fp(batch_data)
        res = self.collate_fn([batch_processed_data])
        return res


class CruiseKVFeatureProvider:
    def __init__(self, **fp_conf):
        from addict import Dict

        fp_conf = Dict(fp_conf)
        fp_type = fp_conf.pop("type")
        save_extra = fp_conf.get("save_extra", False)
        eval_mode = fp_conf.get("eval_mode", False)
        trace_mode = fp_conf.get("trace_mode", False)
        fp = build_feature_provider(fp_type, fp_conf, save_extra, eval_mode, trace_mode)
        self.process_fn = fp.process

    def __call__(self, data):
        return self.process_fn(data)


class CruiseKVBatchFeatureProvider:
    def __init__(self, **fp_conf):
        from addict import Dict

        fp_conf = Dict(fp_conf)
        fp_type = fp_conf.pop("type")
        save_extra = fp_conf.get("save_extra", False)
        eval_mode = fp_conf.get("eval_mode", False)
        trace_mode = fp_conf.get("trace_mode", False)
        fp = build_feature_provider(fp_type, fp_conf, save_extra, eval_mode, trace_mode)
        if hasattr(fp, "collate"):
            self._collate_fn = fp.collate
        else:
            self._collate_fn = default_collate

    def __call__(self, batch_data):
        return self._collate_fn(batch_data)


class CruiseFakeProcessor:
    def __call__(self, data):
        return data


__all__ = [
    "EcomLiveGandalfParquetAutoDisCruiseDataModule",
    "EcomLiveGandalfParquetAutoDisFeatureProvider",
]
