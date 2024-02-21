# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-02-27 20:03:19
# Modified: 2023-02-27 20:03:19
import json
import logging
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import requests
import torch
import yaml
from utils.driver import get_logger


def async_run(func, inputs, pool_size=16, use_thread=True, multi_args=False):
    result = []
    exe = ThreadPoolExecutor if use_thread else ProcessPoolExecutor
    with exe(max_workers=pool_size) as executor:
        if multi_args:
            for args in inputs:
                task = executor.submit(func, *args)
                result.append(task.result())
        else:
            for res in executor.map(func, inputs):
                result.append(res)
    return result


def count_params(model):
    num_params = 0
    trainable_params = 0
    for param in model.parameters():
        num_params += np.prod(param.shape)
        if param.requires_grad:
            trainable_params += np.prod(param.shape)
    get_logger().info(
        "Number of parameters: {:.2f}M, trainable parameters: {:.2f}M".format(
            num_params / 10**6, trainable_params / 10**6
        )
    )


def merge_into_target_dict_backup(src_dict, sub_dict, prefix, is_loss_item=False):
    for key, val in sub_dict.items():
        if is_loss_item:
            if isinstance(val, torch.Tensor):
                src_dict[f"{prefix}_{key}"] = val.mean().item()
            if isinstance(val, float):
                src_dict[f"{prefix}_{key}"] = val
        else:
            if isinstance(val, torch.Tensor):
                if len(val.shape) >= 1:
                    src_dict[f"{prefix}_{key}"] = val.mean().item()
                else:
                    src_dict[f"{prefix}_{key}"] = val.item()  # diff
            if isinstance(val, float):
                src_dict[f"{prefix}_{key}"] = val


def merge_into_target_dict(src_dict, sub_dict, is_loss_item=False):
    for key, val in sub_dict.items():
        if is_loss_item:
            if isinstance(val, torch.Tensor):
                src_dict[f"{key}"] = val.mean().item()
            if isinstance(val, float):
                src_dict[f"{key}"] = val
        else:
            if isinstance(val, torch.Tensor):
                if len(val.shape) >= 1:
                    src_dict[f"{key}"] = val.mean().item()
                else:
                    src_dict[f"{key}"] = val.item()  # diff
            if isinstance(val, float):
                src_dict[f"{key}"] = val


def load_conf(json_path):
    with open(json_path) as f:
        config = json.load(f)
    return config


def load_from_yaml(yaml_path):
    with open(yaml_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            config = {}
        return config


def load_from_tcc(tcc_key, tcc_psm="ecom.govern.live_gandalf"):
    import bytedtcc

    tcc_client = bytedtcc.ClientV2(tcc_psm, "default")
    conf_str = tcc_client.get(tcc_key)
    return json.loads(conf_str)


def load_from_bbc(bbc_key, bbc_psm="ecom.govern.algo"):
    bbc_client = BbcClient(bbc_psm)
    conf_str = bbc_client.read_config(bbc_key)["content"]
    return json.loads(conf_str)


def check_config(config: Dict, enable_train: bool, enable_test: bool, trace_model: bool):
    if enable_train:
        # '--enable_train --enable_test' means test from the
        # best checkpoint produced during train phase
        if enable_test:
            config.tester.from_best_checkpoint = True
        # '--enable_train --trace_model' means trace from the
        # best checkpoint produced during train phase
        if trace_model:
            config.tracer.from_best_checkpoint = True
    else:
        if enable_test and not config.tester.resume_checkpoint:
            get_logger().warning(
                "Have NOT specified resume checkpoint for test phase, " "will automatically search best checkpoint"
            )


def update_config(config: Dict, config_override: List[str], delimiter: str = "=") -> None:
    """update config using extra args in format like 'a.b.c=1'"""

    def _key_has_index(sub_key):
        # if (has_index := re.search(r"\[([0-9]+)\]", sub_key)) is not None:
        has_index = re.search(r"\[([0-9]+)\]", sub_key)
        if has_index is not None:
            list_index = has_index.group(1)
            real_sub_key = re.search(r"(.*)\[", sub_key).group(1)
            return real_sub_key, int(list_index)
        else:
            return None

    def _override_config(keys: List[str]) -> Tuple[Dict, str]:
        cfg, conf_repr = config, "config"
        sub_key = None
        for idx, sub_key in enumerate(keys):
            # if (split_sub_key := _key_has_index(sub_key)) is not None:
            split_sub_key = _key_has_index(sub_key)
            if split_sub_key is not None:
                real_sub_key, list_index = split_sub_key
                if not isinstance(cfg, Dict) or real_sub_key not in cfg:
                    raise ValueError(f'undefined key "{real_sub_key}" detected for "{conf_repr}"')
                list_cfg = cfg.get(real_sub_key)
                if not isinstance(list_cfg, List) or list_index >= len(list_cfg):
                    raise ValueError(
                        f'Not is List or Index {list_index} out of bound detected for "{conf_repr}.{real_sub_key}"'
                    )
                if idx < len(keys) - 1:
                    cfg = list_cfg[list_index]
                    conf_repr += f".{sub_key}"
            else:
                if not isinstance(cfg, Dict) or sub_key not in cfg:
                    raise ValueError(f'undefined key "{sub_key}" detected for "{conf_repr}"')
                if idx < len(keys) - 1:
                    cfg = cfg.get(sub_key)
                    conf_repr += f".{sub_key}"
        return cfg, sub_key

    for cfg in config_override:
        item = cfg.split(delimiter, 1)
        assert len(item) == 2, "Format Error: must be key=value: " + cfg
        key, value = item
        obj, leaf = _override_config(key.split("."))
        if value in ["True", "False", "true", "false"]:
            obj[leaf] = eval(value)
        else:
            if obj[leaf] is not None and (
                isinstance(obj[leaf], int)
                or isinstance(obj[leaf], str)
                or isinstance(obj[leaf], float)
                or isinstance(obj[leaf], bool)
            ):
                obj[leaf] = type(obj[leaf])(value)
            else:
                obj[leaf] = eval(value)
        get_logger().info(f"Update config key [{leaf}] with value: {obj[leaf]}")


def init_seeds(seed=42, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True  # find suitable cudnn algo
    if cuda_deterministic:  # slower but reproducible
        torch.backends.cudnn.deterministic = True

    else:  # faster but no reproducible
        torch.backends.cudnn.deterministic = False


class BbcClient(object):
    BBC_CREDENTIAL = {
        "app_id": "45c162e0-90f9-4c3c-874c-f34b1a4d6d77",
        "app_secret": "cb4bd846abebc2a54e751a6e7bfc967d",
    }

    BBC_ERROR_CODES = {
        1: "ParamErr",
        2: "DBError",
        3: "ConfigCreateFailed",
        4: "ConfigUpdateFailed",
        5: "CallbackFailed",
        6: "FormatErr",
        7: "ConfigNotFound",
    }

    def __init__(self, tenant, key="gandalf_threshold_map"):
        self._tenant_url = "https://horizon.bytedance.net/oapi/bbc/config/" + tenant
        self._access_token = self.get_horizon_token()

    def get_horizon_token(self):
        for _ in range(2):
            rsp = requests.post(
                "https://horizon.bytedance.net/api/horizon/auth/accessToken",
                json=self.BBC_CREDENTIAL,
                timeout=30,
            )
            if not rsp.status_code == requests.codes.ok:  # Retry
                time.sleep(5)
            else:
                return rsp.json()["access_token"]
        raise Exception("访问 Horizon Open API 失败，错误[%d]为：%s" % (rsp.status_code, rsp.text))

    def push_online(self, value, key):
        # 额外构造bbc content
        content = {"default": {"threshold_map": value}}

        result = self.read_config(key)
        prevExist = True if result else False

        params = {
            "config_key": key,
            "content": json.dumps(content),
            "version": result["version"] if prevExist else 0,
            "config_comment": "created by maxwell",
        }
        if not result:
            params["tags"] = '["reckon"]'
            params["config_type"] = 1  # JSON
        for _ in range(2):
            try:
                self._write_config(params, prevExist=prevExist)
                return
            except Exception as e:
                logging.info("Write config key[%s] failed, reason: [%s]", key, e)
        raise Exception("上传meta到bbc失败，请检查配置与日志！ ")

    def push_conf_online(self, content, key):
        result = self.read_config(key)
        prevExist = True if result else False

        params = {
            "config_key": key,
            "content": json.dumps(content),
            "version": result["version"] if prevExist else 0,
            "config_comment": "created by maxwell",
        }
        if not result:
            params["tags"] = '["reckon"]'
            params["config_type"] = 1  # JSON
        for _ in range(2):
            try:
                self._write_config(params, prevExist=prevExist)
                return
            except Exception as e:
                logging.info("Write config key[%s] failed, reason: [%s]", key, e)
        raise Exception("上传本地配置到bbc失败，请检查配置与日志！ ")

    def read_config(self, key):
        for _ in range(2):
            rsp = requests.get(
                self._tenant_url,
                params={"config_key": key},
                headers={"Token": self._access_token},
                timeout=30,
            )
            if not rsp.status_code == requests.codes.ok:  # Retry
                time.sleep(5)
            else:
                rsp = rsp.json()
                code = rsp["code"]
                if code == 200:  # Success
                    return rsp["data"]
                elif code == 7:  # ConfigNotFound
                    return {}
                else:
                    message = self.BBC_ERROR_CODES.get(code, "UnknownError")
                    raise Exception("访问 BBC API 失败，错误为：%s" % message)
        raise Exception("访问 BBC API 失败，错误[%d]为：%s" % (rsp.status_code, rsp.text))

    def _write_config(self, params, prevExist=True):
        rpc_func = requests.put if prevExist else requests.post
        rsp = rpc_func(
            self._tenant_url,
            json=params,
            headers={"Token": self._access_token},
            timeout=30,
        )
        if not rsp.status_code == requests.codes.ok:  # Retry
            time.sleep(5)
            rsp = rpc_func(
                self._tenant_url,
                json=params,
                headers={"Token": self._access_token},
                timeout=30,
            )
        if not rsp.status_code == requests.codes.ok:
            raise Exception("访问 BBC API 失败，错误[%d]为：%s" % (rsp.status_code, rsp.text))
        code = rsp.json()["code"]
        if code != 200:
            message = self.BBC_ERROR_CODES.get(code, "UnknownError")
            raise Exception("访问 BBC API 失败，错误为：%s" % message)
