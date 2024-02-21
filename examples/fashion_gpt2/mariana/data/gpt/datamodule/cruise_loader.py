import json
import math
import os
import pickle
import random
import warnings

import requests
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import all_gather_object, barrier, broadcast_object_list
from torch.utils.data import _utils

try:
    from torch.utils.data.dataloader import BytedDataLoader as DataLoader
except ImportError:
    warnings.warn(
        "Unable to import BytedDataLoader from byted-torch, " "some dataloading acceleration option is disabled"
    )
    from torch.utils.data import DataLoader

from typing import Callable, Dict, List, Optional, Union

from cruise.data_module import hdfs_utils
from cruise.data_module.cruise_kv_dataset import download_kv_index_file
from cruise.data_module.cruise_kv_dataset import get_kv_keys as get_kv_keys_base
from cruise.data_module.cruise_parquet_dataset import get_hdfs_block_size, get_hdfs_host
from cruise.data_module.cruise_parquet_dataset import get_parquet_length as get_parquet_length_single
from cruise.data_module.cruise_parquet_dataset import pf
from cruise.data_module.cruise_tfidx_dataset import get_tfrecord_length as get_tfrecord_length_by_idx
from cruise.data_module.cruise_tfrecord_dataset import get_tfrecord_length

# from cruise.data_module.hybrid_dataset import HybridDataset
from cruise.data_module.gpu_wrapper import GPULoader
from cruise.data_module.utils import (
    LazyLoader,
    download_hdfs_data,
    get_dataset_size,
    get_free_disk_space,
    query_files_length_from_rh2,
    set_native_hdfs_security_permission,
    tf_hdfs_init,
    tf_sample_sharding_is_enabled,
    update_file_lengths_to_rh2,
    use_native_hdfs,
)
from cruise.utilities.rank_zero import once_only
from cruise.utilities.report_usage import report_usage

from .hybrid_dataset import HybridDataset

tf = LazyLoader("tf", globals(), "tensorflow")
pq = LazyLoader("pq", globals(), "pyarrow.parquet")
byted_dataloader = LazyLoader("dataloader", globals(), "dataloader")

parquet_ds_length_info = dict()
kv_ds_length_info = dict()


__all__ = ["DistributedCruiseDataLoader"]


@once_only
def report_data_module():
    report_usage("data_module")


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world():
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def cruise_broadcast(value):
    if isinstance(value, list):
        broadcast_tmp = value
    else:
        broadcast_tmp = [value]
    # TODO: may need device
    broadcast_object_list(broadcast_tmp)
    return broadcast_tmp


def get_kv_keys(url):
    download_kv_index_file(url)
    keys = get_kv_keys_base(url)
    return keys


def get_kv_length(url):
    global kv_ds_length_info
    if url not in kv_ds_length_info:
        keys = get_kv_keys(url)
        kv_ds_length_info[url] = len(keys)
    return kv_ds_length_info[url]


def get_pickle_length(urls):
    if use_native_hdfs():
        set_native_hdfs_security_permission()
    fs = pf.HadoopFileSystem(host=get_hdfs_host(), port=0, buffer_size=get_hdfs_block_size())
    total_length = 0
    for path in urls:
        with fs.open_input_file(path) as f:
            meta_dict = pickle.loads(f.read())
            total_length += len(meta_dict)
    return total_length


# This is an easy way for detecting source type
def detect_source_type(data_source):
    file_example = data_source[0]
    if "parquet" in file_example:
        return "parquet"
    if "tfrecord" in file_example:
        return "tfrecord"
    key = get_kv_keys(file_example)
    if key:
        return "kv"
    else:
        return "jsonl"


def get_files_length(urls, file_type):
    rst = []
    if file_type == "parquet":
        return get_parquet_length(urls)
    elif file_type == "tfrecord":
        rst = get_tfrecord_length(urls)
    elif file_type == "kv":
        for url in urls:
            rst.append(get_kv_length(url))
    elif file_type == "jsonl":
        for url in urls:
            length = 0
            with hdfs_utils.hdfs_open(url) as f:
                for _ in f:
                    length += 1
            rst.append(length)
    elif file_type == "tfidx":
        for url in urls:
            rst.append(get_tfrecord_length_by_idx(url))
    else:
        raise RuntimeError(f"not supported type: {file_type}")

    return rst


def get_parquet_length(urls):
    global parquet_ds_length_info
    if not isinstance(urls, List):
        urls = [urls]
    new_urls = [i for i in urls if i not in parquet_ds_length_info]
    rh2_cache_lens = query_files_length_from_rh2(new_urls)
    need_length = []
    for i, url in enumerate(new_urls):
        if rh2_cache_lens[i] <= 0:
            need_length.append(url)
        else:
            parquet_ds_length_info[url] = rh2_cache_lens[i]
    if need_length:
        with mp.Pool(20) as p:
            new_res = p.map(get_parquet_length_single, need_length)
        post_len_list = []
        for url, length in zip(need_length, new_res):
            parquet_ds_length_info[url] = length
            post_len_list.append({"hdfs_file": url, "samples": length})
        update_file_lengths_to_rh2(post_len_list)
    res = [parquet_ds_length_info[i] for i in urls]
    return res


def get_length_from_sources(data_source, data_type, dist_flag=True):
    files_len = []
    files_name = []
    if dist_flag:
        rank = get_rank()
        world = get_world()
        rank_data_source = data_source[rank::world]
    else:
        rank_data_source = data_source

    rh2_cache_lens = query_files_length_from_rh2(rank_data_source)
    post_len_list = []
    no_cache_files = []
    for i, file_path in enumerate(rank_data_source):
        if rh2_cache_lens[i] <= 0:
            no_cache_files.append(file_path)
        else:
            files_len.append(rh2_cache_lens[i])
            files_name.append(file_path)

    new_res = get_files_length(no_cache_files, data_type)
    for url, length in zip(no_cache_files, new_res):
        files_name.append(url)
        files_len.append(length)
        post_len_list.append({"hdfs_file": url, "samples": length})

    if post_len_list:
        update_file_lengths_to_rh2(post_len_list)
    if dist_flag:
        allgather_rst = [None] * world
        all_gather_object(allgather_rst, files_len)
        allgather_name_rst = [None] * world
        all_gather_object(allgather_name_rst, files_name)
        total_lens = []
        total_paths = []
        for each in allgather_rst:
            total_lens.extend(each)
        for each in allgather_name_rst:
            total_paths.extend(each)
    else:
        total_lens = files_len
        total_paths = files_name

    meta = {}
    for length, path in zip(total_lens, total_paths):
        meta[path] = length
    return sum(total_lens), meta


def shard_source(
    source,
    rank,
    world,
    num_workers,
    source_type,
    source_meta,
    drop_last=True,
    length=-1,
    batch_size=-1,
):
    if num_workers == 0:
        num_workers = 1
    if source_type == "kv":
        # since kv dataset would handle the sharding between num workers by itself, here we set num_worker = 1
        return (
            shard_source_by_sample(source, rank, world, 1, source_meta, source_type, drop_last),
            0,
            1,
        )
    # use sample sharding for all parquet dataset
    elif source_type == "parquet" or (source_type == "tfrecord" and tf_sample_sharding_is_enabled()):
        res = []
        if not source_meta:
            # source_meta = [get_parquet_length(url) for url in source]
            if source_type == "parquet":
                source_meta = get_parquet_length(source)
            else:
                _, source_meta = get_length_from_sources(source, "tfrecord", world > 1)
                source_meta = [source_meta[i] for i in source]
        for i in range(num_workers):
            # here we have two ways to calculate the new rank (process rank, not gpu rank)
            # 1: rank + i * world
            # 2: rank * num_workers + i
            # we choose the method 1 since this way for the shard results,
            # each gpu's dataloading processes would see data from different data sources
            # while if we choose method 2, each gpu's dataloading processes would see
            # data from single or conjunctive datasets
            res.append(
                shard_source_by_sample(
                    source,
                    rank + i * world,
                    world,
                    num_workers,
                    source_meta,
                    source_type,
                    drop_last,
                    batch_size,
                    length,
                )
            )
        return res, 0, 1
    elif source_type == "tfidx":
        if not source_meta:
            source_meta = [get_tfrecord_length_by_idx(url) for url in source]
        return (
            shard_source_by_sample(source, rank, world, 1, source_meta, source_type, drop_last),
            0,
            1,
        )
    # length=-1 means reading over all the data only once, thus we need to
    # apply the sample sharding here to make the data evenly sharded
    elif source_type == "tfrecord" and length == -1:
        return source, rank, world
    else:
        if len(source) < world:
            if source_type == "tfrecord":
                return source, rank, world
            else:
                raise ValueError("File number is smaller than num replicas, cannot shard parquet/jsonl files.")
        return source[rank::world], 0, 1


def shard_source_by_sample(
    source,
    rank,
    world,
    num_workers,
    source_lens,
    src_type="kv",
    drop_last=True,
    batch_size=-1,
    target_step=-1,
):
    if not source_lens:
        source_lens = []
        if src_type == "kv":
            for url in source:
                source_lens.append(len(get_kv_keys(url)))
        elif src_type == "parquet":
            source_lens = get_parquet_length(source)
    total_lens = sum(source_lens)

    if total_lens < batch_size * world and target_step > 0:
        # if the data size is too small (less than one single batch),
        # and target_step != -1 (-1 means to read the data exactly once)
        # data would be generated repeatedly to meet the target step
        # we set data_per_rank to be exactly batch_size
        # and repeat the sources large enough times to make it successfully go through the following sharding logic
        data_per_rank = batch_size
        repeat_n = (batch_size * world * num_workers + total_lens - 1) // total_lens
        source = source * repeat_n
        source_lens = source_lens * repeat_n
        # since we have repeated the data, we do not need to care about drop_last attribute,
        # just set remain to be 0
        remain = 0
    else:
        data_per_rank = total_lens // (world * num_workers)
        remain = total_lens - data_per_rank * (world * num_workers)

    start_ps_idx = 0
    s = 0
    splits = []
    total_start_idx = 0
    i, size = 0, 0
    for i, url in enumerate(source):
        size = source_lens[i]
        while s <= total_start_idx < s + size:
            start_ps_idx = i
            start_iter_idx = total_start_idx - s
            splits.append((start_ps_idx, start_iter_idx))
            total_start_idx += data_per_rank
            # here we handle drop_last = False
            # add one more sample to current shard when there is residual
            if not drop_last and remain > 0:
                total_start_idx += 1
                remain -= 1
        s += size
    # if current rank does not have a shard,
    # which might be possible when data source is short and target_step = -1
    # we return empty list
    if len(splits) <= rank:
        assert target_step == -1
        return []

    splits.append((i, size))

    start_ps_idx, start_iter_idx = splits[rank]
    end_ps_idx, end_iter_idx = splits[rank + 1]
    # happen to reach the first of a dataset
    if end_iter_idx == 0:
        end_ps_idx -= 1
        end_iter_idx = source_lens[end_ps_idx]

    path_start_end_info, overall_iters = [], 0
    for ps_idx in range(start_ps_idx, end_ps_idx + 1):
        tmp_start_idx = 0 if ps_idx != start_ps_idx else start_iter_idx
        if ps_idx != end_ps_idx:
            tmp_end_idx = source_lens[ps_idx]
        else:
            tmp_end_idx = end_iter_idx
        overall_iters += tmp_end_idx - tmp_start_idx
        path_start_end_info.append((source[ps_idx], tmp_start_idx, tmp_end_idx))
    return path_start_end_info


def simple_collate(x):
    return x


def get_fake_gpu_trans():
    return simple_collate


class DistributedCruiseDataLoader:
    r"""
    Cruise Data Loader, an user-friendly data loader, support hybrid multiple tasks.

        Args:
            parquet_cache_on (bool, optional):
                Enable parquet dataset caching mode. Defaults to True.
            no_sharding (bool, optional):
                Do not perform dataset sharding. Defaults to False.
            fast_resume (bool, optional):
                resume the loader state in a faster way, which might sacrifice the resume precision
            synthetic_sample (bool, optional):
                if True, the prefetch processes will use fist sample as synthetic data and skip
                the following data reading process. That data will be transformed/processed later
                within whole pipeline.
            shuffle_buffer_size (int, optional, default=100): if `shuffle` is True and loader loading on iterable Datasets,
                i.e Parquet Dataset or TFRecord Dataset, these Datasets will open a sample buffer to shuffle data,
                the size of that sample buffer is equal to `shuffle_buffer_size`.
            dyn_bsz(bool, default=False): whether to use dynamic batchsize to fully utilize mem.
            dyn_bsz_margin (float, default=0.0): determine when the collection of batch is done. Larger dyn_bsz_margin, the more mem is consumed and more data is sampled into the batch.

    """

    def __init__(
        self,
        data_sources: List[List],
        keys_or_columns: List[List[str]],
        batch_sizes: List,
        num_workers: int,
        num_readers: Optional[Union[int, List[int]]],
        decode_fn_list: List[Callable],
        processor,
        predefined_steps: Union[int, str] = None,
        source_types: Optional[List[str]] = None,
        seed=0,
        last_step: int = 0,
        kv_loader_state: Dict = {},
        shuffle: bool = False,
        task_id_list: List = [],
        use_gpu: bool = False,
        enable_borrower: bool = False,
        multiplex_weights: list = [],
        use_all_gather: bool = False,
        remain_sample_idx: bool = False,
        transform_output_many: bool = False,
        drop_last: bool = True,
        key_mapping: List[Dict] = None,
        local_fetch_path: str = None,
        pin_memory: bool = True,
        parquet_cache_on: bool = True,
        use_arnold: bool = True,
        no_sharding: bool = False,
        shuffle_buffer_size: int = 100,
        synthetic_sample: bool = False,
        synthetic_batch: bool = False,
        fast_resume: bool = False,
        dyn_bsz: bool = False,
        dyn_bsz_margin: float = 0.0,
        num_warmup_steps: int = -1,
        **kwargs,
    ):
        report_data_module()

        self.data_sources = data_sources
        if not task_id_list:
            self.task_id_list = [None for _ in data_sources]
        else:
            assert len(task_id_list) == len(data_sources), "task_id_list should have equal length as data_sources"
            self.task_id_list = task_id_list
        self.batch_sizes = batch_sizes
        self.num_workers = num_workers
        self.num_readers = num_readers
        self.source_types = source_types
        self.return_keys = keys_or_columns
        self.enable_borrower = enable_borrower
        self.multiplex_weights = multiplex_weights
        self.source_meta = {}
        self.use_all_gather = use_all_gather
        self.remain_sample_idx = remain_sample_idx
        self.transform_output_many = transform_output_many
        self.drop_last = drop_last
        self.no_sharding = no_sharding
        self.fast_resume = fast_resume
        self.dyn_bsz = dyn_bsz
        self.dyn_bsz_margin = dyn_bsz_margin
        self.num_warmup_steps = num_warmup_steps
        if self.no_sharding:
            self.rank = 0
            self.world = 1
            if self.num_workers > 1:
                msg = f"Dataloader num_workers adjusted from {self.num_workers} to 1 with `no_sharding=True`"
                warnings.warn(msg)
                self.num_workers = 1
        else:
            self.rank = get_rank()
            self.world = get_world()
        self.kwargs = kwargs
        triplet_sampling = kwargs.get("triplet_sampling", False)
        if triplet_sampling:
            self.length = self._init_triplet_length(predefined_steps)
        else:
            self.length = self._init_length(predefined_steps, drop_last)
        self.resume = True if last_step > 0 else False
        self.reset_resume_step = False
        self.seed = seed
        self.epoch = 0
        self.step = last_step

        # Zhi: bsz is per rank now by removing this
        # for i, batch in enumerate(self.batch_sizes):
        #     self.batch_sizes[i] = batch // self.world
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.use_gpu = use_gpu
        self.transform_fn = processor.transform
        self.batch_transform_fn = processor.batch_transform
        self.post_process = getattr(processor, "post_transform", None)
        self.processor = processor
        self.decode_fn = decode_fn_list
        # only used for arnold dataset
        self.transform_replace_all = kwargs.get("transform_replace_all", False)

        self.repeat = True
        # use length == -1 to indicate loading all the data as one epoch
        if predefined_steps == -1:
            self.repeat = False
        self.stop_queue = mp.Queue()
        if key_mapping:
            self.key_mapping = []
            for each in key_mapping:
                if not each:
                    self.key_mapping.append({})
                    continue
                num_keys = len(each["previous"])
                mappings = {each["previous"][i]: each["current"][i] for i in range(num_keys)}
                self.key_mapping.append(mappings)
        else:
            self.key_mapping = [None] * len(self.data_sources)
        self.pin_memory = pin_memory
        self.parquet_cache_on = parquet_cache_on

        local_fetch_success = False
        if local_fetch_path:
            if not os.path.exists(local_fetch_path):
                os.makedirs(local_fetch_path, exist_ok=True)
            ds_total_size = 0
            for i, data_source in enumerate(data_sources):
                for dataset in data_source:
                    ds_total_size += get_dataset_size(dataset, self.source_types[i])
            available = get_free_disk_space("/opt/tiger")
            if available * 0.5 > ds_total_size / 1024**3:
                dataset_mappings = download_hdfs_data(data_sources, source_types, local_fetch_path)
                barrier()
                local_fetch_success = True
                for i, data_source in enumerate(data_sources):
                    for j, dataset in enumerate(data_source):
                        self.data_sources[i][j] = dataset_mappings[dataset]

        # We found that torch iter_loader is better after fetching data to the local disk
        if int(os.getenv("CRUISE_LOADER_USE_ARNOLD_DATASET", "1")) and not local_fetch_success and use_arnold:
            self.kv_source_idx = [i for i in range(len(self.data_sources)) if self.source_types[i] == "kv"]
            self.iter_source_idx = [i for i in range(len(self.data_sources)) if self.source_types[i] != "kv"]
            self.key_mapping = [
                self.key_mapping[i] for i in range(len(self.data_sources)) if self.source_types[i] != "kv"
            ]
        else:
            self.kv_source_idx = []
            self.iter_source_idx = list(range(len(self.data_sources)))
        self.synthetic_sample = synthetic_sample
        if synthetic_sample:
            warnings.warn(f"Dataloader is runing on synthetic samples, data loading processes is skipped.")
        self.synthetic_batch = synthetic_batch
        if synthetic_batch:
            warnings.warn(f"Dataloader is runing on synthetic batch, transform/batch_transform processes is skipped.")
        self.kv_loader = None
        self.torch_iter = None
        self.kv_loader_state = kv_loader_state
        self.torch_loader = None
        if not self.shuffle and self.iter_source_idx:
            # create a loader instance when shuffle is False to save time
            self.torch_loader = self._create_iter_loader()

        if self.kv_source_idx:
            self._create_kv_loader()

        self.loader_prob = []
        if self.multiplex_weights:
            kv_weights = sum([self.multiplex_weights[i] for i in self.kv_source_idx])
            iter_weights = 1 - kv_weights
            if kv_weights:
                self.loader_prob.append(kv_weights)
            if iter_weights:
                self.loader_prob.append(iter_weights)
        self.ask_subprocess_stop = False

    def _create_kv_loader(self):
        kv_sources = [self.data_sources[i] for i in self.kv_source_idx]
        kv_batch_sizes = [self.batch_sizes[i] for i in self.kv_source_idx]
        # for arnold dataset, multiple kv dataset use the same kv num_readers, here we just get the maximum
        kv_num_readers = max([self.num_readers[i] for i in self.kv_source_idx])
        kv_task_id_list = [self.task_id_list[i] for i in self.kv_source_idx]
        kv_ds_split_num = self.kwargs.get("dataset_split_num", 4)
        kv_epochs_for_reader = self.kwargs.get("epochs_for_reader", 5)
        kv_decode_fn = None if not self.decode_fn else [self.decode_fn[i] for i in self.kv_source_idx]
        kv_return_keys = None if not self.return_keys else [self.return_keys[i] for i in self.kv_source_idx]
        triplet_sampling = self.kwargs.get("triplet_sampling", False)
        triplet_meta_dict_path = self.kwargs.get("triplet_meta_dict_path", "")
        triplet_meta_dict_format = self.kwargs.get("triplet_meta_dict_format", "pickle")
        triplet_p = self.kwargs.get("triplet_p", 1)
        triplet_k = self.kwargs.get("triplet_k", 1)
        from .arnold_dataset import ArnoldDataset

        size_guarantee = not self.shuffle
        self.kv_loader = ArnoldDataset(
            kv_sources,
            kv_batch_sizes,
            kv_task_id_list,
            self.num_workers,
            kv_num_readers,
            world_size=self.world,
            rank=self.rank,
            shuffle=self.shuffle,
            return_keys=kv_return_keys,
            decode_fn=kv_decode_fn,
            trans_fn=(
                self.transform_fn,
                self.batch_transform_fn,
                self.post_process,
            ),
            dataset_split_num=kv_ds_split_num,
            pin_memory=self.pin_memory,
            epochs_for_reader=kv_epochs_for_reader,
            remain_sample_idx=self.remain_sample_idx,
            resume_state=self.kv_loader_state,
            transform_replace_all=self.transform_replace_all,
            triplet_info=(
                triplet_sampling,
                triplet_meta_dict_path,
                triplet_meta_dict_format,
                triplet_p,
                triplet_k,
            ),
            size_guarantee=size_guarantee,
            synthetic_sample=self.synthetic_sample,
            transform_many=self.transform_output_many,
        )

    def _create_iter_loader(self):
        shard_data, shard_rank_info = self.shard_data_sources()
        iter_sources = [shard_data[i] for i in self.iter_source_idx]
        iter_shard_rank_info = [shard_rank_info[i] for i in self.iter_source_idx]
        iter_source_types = [self.source_types[i] for i in self.iter_source_idx]
        iter_batch_sizes = [self.batch_sizes[i] for i in self.iter_source_idx]
        iter_num_readers = [self.num_readers[i] for i in self.iter_source_idx]
        iter_task_id_list = [self.task_id_list[i] for i in self.iter_source_idx]
        iter_return_keys = None if not self.return_keys else [self.return_keys[i] for i in self.iter_source_idx]
        iter_decode_fn = None if not self.decode_fn else [self.decode_fn[i] for i in self.iter_source_idx]
        iter_multiplex_weights = (
            [] if not self.multiplex_weights else [self.multiplex_weights[i] for i in self.iter_source_idx]
        )
        if iter_multiplex_weights:  # normalize the probability to make them sum to 1
            weight_sum = sum(iter_multiplex_weights)
            iter_multiplex_weights = [i / weight_sum for i in iter_multiplex_weights]
        batch_shuffle = self.kwargs.get("batch_shuffle", False)
        iter_dataset = HybridDataset(
            iter_sources,
            iter_source_types,
            iter_batch_sizes,
            iter_num_readers,
            iter_return_keys,
            iter_decode_fn,
            (self.transform_fn, self.batch_transform_fn, self.post_process),
            self.shuffle,
            self.seed,
            self.step,
            shard_rank_info=iter_shard_rank_info,
            task_id_list=iter_task_id_list,
            multiplex_weights=iter_multiplex_weights,
            remain_sample_idx=self.remain_sample_idx,
            repeat=self.repeat,
            stop_queue=self.stop_queue,
            key_mapping_list=self.key_mapping,
            drop_last=self.drop_last,
            batch_shuffle=batch_shuffle,
            parquet_cache_on=self.parquet_cache_on,
            shuffle_buffer_size=self.shuffle_buffer_size,
            fast_resume=self.fast_resume,
            synthetic_sample=self.synthetic_sample,
            transform_many=self.transform_output_many,
            dyn_bsz=self.dyn_bsz,
            dyn_bsz_margin=self.dyn_bsz_margin,
            num_warmup_steps=self.num_warmup_steps,
        )
        torch_loader = DataLoader(
            iter_dataset,
            num_workers=self.num_workers,
            collate_fn=simple_collate,
            batch_size=None,
            pin_memory=self.pin_memory,
        )
        if self.enable_borrower and hasattr(torch_loader, "_enable_borrower"):
            torch_loader.enable_borrower()
        return torch_loader

    def __iter__(self):
        """Get the dataset iterator."""
        if self.stop_queue is not None and not self.stop_queue.empty():
            self.stop_queue.get()
        self.ask_subprocess_stop = False
        if self.resume:
            self.resume = False

            # Reconstruct self.torch_loader at the beginning of the next epoch to reset resume_step.
            self.reset_resume_step = True
        else:
            if self.shuffle:
                self._shuffle()
            self.step = 0

            if self.reset_resume_step:
                if not self.shuffle and self.iter_source_idx:
                    self.torch_loader = self._create_iter_loader()
                self.reset_resume_step = False

        if self.iter_source_idx:
            if self.shuffle or self.torch_loader is None:
                torch_loader = self._create_iter_loader()
            else:
                torch_loader = self.torch_loader
            self.torch_iter = iter(torch_loader)
        self.data_iters = []
        if self.kv_loader is not None:
            self.data_iters.append(self.kv_loader)
        if self.torch_iter is not None:
            self.data_iters.append(self.torch_iter)

        # TODO add gpu loader
        # if not self.use_gpu:
        #     self.loader = iter(torch_loader)
        # else:
        #     # right now just use fake_gpu_trans as a place holder
        #     gpu_loader = GPULoader(torch_loader, get_fake_gpu_trans, step=self.length)
        #     self.loader = iter(gpu_loader)
        return self

    def combine_two_loader(self, data):
        item1 = data[0]
        item2 = data[1]
        if isinstance(item1, list):
            return item1 + item2
        if isinstance(item1, dict):
            res = {}
            for k in item1.keys():
                res[k] = self.combine_two_loader([item1[k], item2[k]])
            return res

    def _get_data_from_loaders(self):
        if self.multiplex_weights:
            data_iter = random.choices(self.data_iters, weights=self.loader_prob, k=1)[0]
            return next(data_iter)
        else:
            if self.kv_loader is not None:
                kv_data = next(self.kv_loader)
                if not self.torch_iter:
                    return kv_data
            if self.torch_iter is not None:
                iter_data = next(self.torch_iter)
                if not self.kv_loader:
                    return iter_data
            data = self.combine_two_loader([kv_data, iter_data])
            return data

    def __next__(self):
        # encase self.length = -1
        if self.length > 0 and self.step >= self.length:
            # if self.use_gpu:
            #     # call __next__ one more time to actually stop gpu loader
            #     next(self.loader)
            self.stop_queue.put(1)
            self.ask_subprocess_stop = True
            if self.torch_iter is not None and self.num_workers > 0:
                self.shutdown_torch_iter()
            self.step = 0
            raise StopIteration
        # update step only after successfully getting the data
        # otherwise, the recorded step count might be larger than the actual iterated step
        try:
            if self.synthetic_batch and hasattr(self, "_first_batch"):
                # get the first batch of data only, data will not be processed/transformed
                data = self._first_batch
            else:
                data = self._get_data_from_loaders()
                if not hasattr(self, "_first_batch"):
                    self._first_batch = data
            self.step += 1
            return data
        except StopIteration:
            if self.torch_iter is not None and self.num_workers > 0:
                # when we reach here, dataset processes are already joined/exited
                del self.torch_iter
            self.step = 0
            raise StopIteration

    def __len__(self):
        """Get the dataset length."""
        return self.length

    def shard_data_sources(self):
        shard_data = []
        shard_rank_info = []
        for i, data_source in enumerate(self.data_sources):
            if (
                self.source_types[i] == "kv" and self.kv_source_idx
            ):  # kv type data would use arnold dataset to loader data
                shard_data.append(None)
                shard_rank_info.append((0, 0))
            else:
                source_meta = self.source_meta[i] if self.source_meta else None
                source_data, shard_rank, shard_world = shard_source(
                    data_source,
                    self.rank,
                    self.world,
                    self.num_workers,
                    self.source_types[i],
                    source_meta,
                    self.drop_last,
                    self.length,
                    self.batch_sizes[i],
                )
                shard_data.append(source_data)
                shard_rank_info.append((shard_rank, shard_world))
        return shard_data, shard_rank_info

    def _init_length(self, predefined_steps, drop_last=True):
        if predefined_steps and isinstance(predefined_steps, int) and predefined_steps > 0:
            # Users have predefined lengths for all data sources
            return predefined_steps
        else:
            num_steps = []
            for k, data_source in enumerate(self.data_sources):
                dist_flag = self.world > 1 and self.use_all_gather
                source_len, source_meta = get_length_from_sources(data_source, self.source_types[k], dist_flag)
                source_meta = [source_meta[key] for key in data_source]
                self.source_meta[k] = source_meta
                global_bsz = self.batch_sizes[k] * self.world
                if self.source_types[k] != "parquet":
                    if drop_last:
                        num_steps.append(source_len // global_bsz)
                    else:
                        num_steps.append(math.ceil(source_len / global_bsz))
                else:
                    num_worker = self.num_workers if self.num_workers > 0 else 1
                    data_per_procs = [source_len // (self.world * num_worker)] * num_worker
                    remain = source_len - self.world * num_worker * data_per_procs[0] if not drop_last else 0
                    remain = (remain + self.world - 1) // self.world
                    for i in range(remain):
                        data_per_procs[i] += 1
                    batch_per_procs = [(i + self.batch_sizes[k] - 1) // self.batch_sizes[k] for i in data_per_procs]
                    num_steps.append(sum(batch_per_procs))

            if predefined_steps == "min":
                return min(num_steps)
            else:
                return max(num_steps)

    def _init_triplet_length(self, predefined_steps):
        meta_dict_path = self.kwargs.get("triplet_meta_dict_path", "")
        meta_dict_format = self.kwargs.get("triplet_meta_dict_format", "pickle")
        batch_category = self.kwargs.get("triplet_p", 1)
        if predefined_steps and isinstance(predefined_steps, int) and predefined_steps > 0:
            # Users have predefined lengths for all data sources
            return predefined_steps
        if isinstance(meta_dict_path, str):
            meta_dict_path = [meta_dict_path]
        category_num = 0
        if meta_dict_format == "kv":
            for path in meta_dict_path:
                keys = get_kv_keys(path)
                category_num += len(keys)
        elif meta_dict_format == "pickle":
            ctx = mp.get_context("spawn")
            with ctx.Pool(1) as p:
                category_num = p.map(
                    get_pickle_length,
                    [
                        meta_dict_path,
                    ],
                )[0]
        else:
            raise RuntimeError("triplet dataset format must be kv or pickle, please check the input")

        if self.drop_last:
            return category_num // (batch_category * self.world)
        else:
            return math.ceil(category_num / (batch_category * self.world))

    def _shuffle(self):
        # broadcast seed will occasionally cause hang
        # if get_world() > 1:
        #     cruise_broadcast(self.seed)
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1  # incre seed so each time shuffle will have different order
        for i, data_source in enumerate(self.data_sources):
            if self.source_meta:
                tmp = list(zip(self.data_sources[i], self.source_meta[i]))
                rng.shuffle(tmp)
                self.data_sources[i], self.source_meta[i] = zip(*tmp)
                if isinstance(self.data_sources[i], tuple):
                    self.data_sources[i] = list(self.data_sources[i])
            else:
                rng.shuffle(data_source)

    def __getstate__(self):
        d = self.__dict__.copy()
        if self.kv_loader is not None:
            d["kv_loader_state"] = self.kv_loader.state
        if "torch_iter" in d:
            del d["torch_iter"]
        if "torch_loader" in d:
            del d["torch_loader"]
        if "kv_loader" in d:
            del d["kv_loader"]
        if "data_iters" in d:
            del d["data_iters"]
        if "stop_queue" in d:
            del d["stop_queue"]
        if "transform_fn" in d:
            del d["transform_fn"]
        if "batch_transform_fn" in d:
            del d["batch_transform_fn"]
        if "post_process" in d:
            del d["post_process"]
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.rank = get_rank()
        self.world = get_world()
        self.kv_loader = None
        self.torch_iter = None
        self.torch_loader = None
        self.transform_fn = self.processor.transform
        self.batch_transform_fn = self.processor.batch_transform
        self.post_process = getattr(self.processor, "post_transform", None)
        if self.kv_source_idx:
            self._create_kv_loader()
        self.resume = True
        self.stop_queue = mp.Queue()

        if not self.shuffle and self.iter_source_idx:
            # create a loader instance when shuffle is False to save time
            self.torch_loader = self._create_iter_loader()

    def restore(self, step, epoch, kv_loader_state={}):
        if step > 0:
            self.resume = True
        self.step = step
        self.epoch = epoch
        self.kv_loader_state = kv_loader_state

    def terminate(self):
        """Shutdown the dataloader and its related threads/processes."""
        # in case users do not trigger stop iteration
        if not self.ask_subprocess_stop:
            self.stop_queue.put(1)
            self.ask_subprocess_stop = True
        if self.torch_iter is not None and self.num_workers > 0:
            self.shutdown_torch_iter()
        if self.kv_loader is not None:
            self.kv_loader.terminate()

    def shutdown_torch_iter(self):
        # Below code is copied from _shutdown_workers method in torch, but extend timeout for join
        python_exit_status = _utils.python_exit_status
        if python_exit_status is True or python_exit_status is None:
            return
        if not self.torch_iter._shutdown:
            self.torch_iter._shutdown = True
            try:
                if hasattr(self.torch_iter, "_pin_memory_thread"):
                    self.torch_iter._pin_memory_thread_done_event.set()
                    self.torch_iter._worker_result_queue.put((None, None))
                    self.torch_iter._pin_memory_thread.join()
                    self.torch_iter._worker_result_queue.cancel_join_thread()
                    self.torch_iter._worker_result_queue.close()

                # Exit workers now.
                self.torch_iter._workers_done_event.set()
                for worker_id in range(len(self.torch_iter._workers)):
                    if self.torch_iter._persistent_workers or self.torch_iter._workers_status[worker_id]:
                        self.torch_iter._mark_worker_as_unavailable(worker_id, shutdown=True)
                for w in self.torch_iter._workers:
                    w.join(timeout=300)
                for q in self.torch_iter._index_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                if self.torch_iter._worker_pids_set:
                    _utils.signal_handling._remove_worker_pids(id(self.torch_iter))
                    self.torch_iter._worker_pids_set = False
                for w in self.torch_iter._workers:
                    if w.is_alive():
                        w.terminate()
