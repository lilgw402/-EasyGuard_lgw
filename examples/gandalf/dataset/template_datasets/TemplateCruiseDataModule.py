# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-02-27 20:36:34
# Modified: 2023-02-27 20:36:34
import copy
import math
import os
from typing import List, Optional

import torch
from cruise.data_module import CruiseDataModule
from cruise.data_module.cruise_loader import DistributedCruiseDataLoader
from cruise.data_module.preprocess.create_preprocess import parse_cruise_processor_cfg
from cruise.data_module.preprocess.decode import TFApiExampleDecode
from utils.dataset_utils.create_config import create_cruise_process_config
from utils.dataset_utils.parse_files import get_ds_path
from utils.driver import get_logger
from utils.registry import DATASETS, FEATURE_PROVIDERS
from utils.torch_util import default_collate


@FEATURE_PROVIDERS.register_module()
class TemplateParquetFeatureProvider:
    def __init__(self):
        super(TemplateParquetFeatureProvider, self).__init__()

    def process(self, data):
        pass

    @staticmethod
    def _process_label(label):
        if int(label) == 1:
            return torch.tensor([1], dtype=torch.float32)
        else:
            return torch.tensor([0], dtype=torch.float32)

    @staticmethod
    def _process_mls_label(verify_status, label_length):  # multiclass label
        if int(verify_status) == 1:
            return torch.tensor([1], dtype=torch.float32)
        else:
            return torch.tensor([0], dtype=torch.float32)

    def batch_process(self, batch_data: List[dict]) -> List[dict]:
        return [x for x in list(map(self.process, batch_data)) if x is not None]

    def split_batch_process(self, batch_data_dict: dict) -> dict:
        batch_data_list = self._split_array(batch_data_dict)
        return self.__call__(batch_data_list)

    def __call__(self, batch_data: List[dict]) -> dict:
        return default_collate(self.batch_process(batch_data))


@DATASETS.register_module()
class TemplateCruiseDataModule(CruiseDataModule):
    def __init__(self):
        super(TemplateCruiseDataModule, self).__init__()
        self.save_hparams()

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self):
        return iter([])

    def val_dataloader(self):
        return iter([])

    def test_dataloader(self):
        return iter([])

    def predict_dataloader(self):
        return iter([])

    def create_cruise_dataloader(
        self,
        cfg,
        data_input_dir,
        data_folder,
        arg_dict,
        mode="val",
        specific_bz=None,
    ):
        arg_dict_cp = copy.deepcopy(arg_dict)
        data_sources, data_types = get_ds_path(
            data_input_dir,
            data_folder,
            arg_dict_cp.type,
            arg_dict_cp.filename_pattern,
            arg_dict_cp.file_min_size,
            arg_dict_cp.group_keys,
            arg_dict_cp.shuffle_files,
        )
        ds_num = len(data_sources)
        drop_last = arg_dict_cp.get("drop_last", True)
        shuffle = arg_dict_cp.get("shuffle", True)
        fast_resume = arg_dict_cp.get("fast_resume", True)
        parquet_cache_on = arg_dict_cp.get("parquet_cache_on", True)
        batch_size = arg_dict_cp.get("batch_size", 128)
        predefined_steps = -1
        use_arnold = True

        if mode == "train":
            predefined_steps = cfg.trainer.train_max_iteration

        if mode == "val":
            drop_last = False
            # trick only in val: half bz to lower mem usage
            if arg_dict_cp.get("batch_size_val", -1) == -1:
                get_logger().info("batch_size_val is not set, use batch_size // 2 as default")
                arg_dict_cp.batch_size_val = arg_dict_cp.batch_size // 2
            batch_size = arg_dict_cp.batch_size_val
            predefined_steps = cfg.trainer.test_max_iteration

        if mode == "test":
            drop_last = False
            shuffle = False
            predefined_steps = cfg.tester.max_iteration

        if mode == "trace":
            if "ParquetDataFactory" in arg_dict_cp.df_type:
                shuffle = False

        # use in trace model
        if specific_bz and isinstance(specific_bz, int):
            batch_size = specific_bz

        num_workers = arg_dict_cp.num_workers
        num_readers = [arg_dict_cp.num_parallel_reads] * ds_num
        multiplex_dataset_weights = arg_dict_cp.multiplex_dataset_weights
        multiplex_mix_batch = arg_dict_cp.multiplex_mix_batch
        is_kv = data_types[0] == "kv"
        if is_kv:
            multiplex_mix_batch = True
            use_arnold = num_workers > 0

        if multiplex_mix_batch:
            # for the case when one batch data is mixed by multiple datasets
            if not multiplex_dataset_weights:
                batch_sizes = [batch_size // ds_num] * ds_num
                remain = batch_size - sum(batch_sizes)
                for i in range(remain):
                    batch_sizes[i] += 1
            else:
                batch_sizes = [math.floor(batch_size * p) for p in multiplex_dataset_weights]
                remain = batch_size - sum(batch_sizes)
                for i in range(remain):
                    batch_sizes[i] += 1
            multiplex_dataset_weights = []
        else:
            # for the case when one batch data is from only single dataset each time,
            # while the dataset is chosen randomly from all the given datasets
            if not multiplex_dataset_weights:
                if ds_num > 1:
                    # read each dataset with equal probability when multiplex_dataset_weights is not given
                    multiplex_dataset_weights = [1 / ds_num] * ds_num
                else:
                    # since we only have one dataset, the multiplex_dataset_weights does not affcet the loading logic
                    # we make it to be an empty list here to match the original logic for single dataset
                    multiplex_dataset_weights = []
            batch_sizes = [batch_size] * ds_num
        process_cfg = create_cruise_process_config(cfg, mode, is_kv)
        cruise_processor = parse_cruise_processor_cfg(process_cfg, "")

        keys_or_columns = []
        last_step = 0
        if ds_num > 1 and predefined_steps == -1:
            predefined_steps = "max"
        # define decode_fn
        if data_types[0] == "tfrecord":
            features = arg_dict_cp.data_schema
            enable_tf_sample_sharding = int(os.getenv("CRUISE_ENABLE_TF_SAMPLE_SHARDING", "0"))
            to_numpy = not enable_tf_sample_sharding
            decode_fn = [TFApiExampleDecode(features=features, key_mapping=dict(), to_numpy=to_numpy)] * ds_num
        elif is_kv and not use_arnold:
            # since kv feature provider would get the index 0 of the given data;
            # while in cruise loader, if reading kv data from
            # torch loader, the output data would be the data itself, not a list.
            # Here we make it a list by using decode fn, to
            # ensure it is runnable, but this might be a little bit hacky.
            decode_fn = [lambda x: [x]] * ds_num
        else:
            decode_fn = None
        # Create DistributedCruiseDataLoader
        loader = DistributedCruiseDataLoader(
            data_sources,
            keys_or_columns,
            batch_sizes,
            num_workers,
            num_readers,
            decode_fn,
            cruise_processor,
            predefined_steps,
            data_types,
            last_step,
            shuffle=shuffle,
            multiplex_weights=multiplex_dataset_weights,
            drop_last=drop_last,
            use_arnold=use_arnold,
            transform_replace_all=is_kv,
            fast_resume=fast_resume,
            parquet_cache_on=parquet_cache_on,
        )
        return loader
