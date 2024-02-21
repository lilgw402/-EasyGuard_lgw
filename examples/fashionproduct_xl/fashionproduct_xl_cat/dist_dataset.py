#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-16 15:25:34
LastEditTime: 2020-11-18 18:12:03
LastEditors: Huang Wenguan
Description: hdfs dataset
"""

import logging
import random
from typing import Any, List

import torch
from cruise.utilities.hdfs_io import hlist_files, hopen
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


class DistLineReadingDataset(IterableDataset):  # pylint: disable=W0223
    """
    iterate a set of folders.
    """

    def __init__(
        self,
        data_path: str,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = False,
        repeat: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size

        self.files = hlist_files(data_path.split(","))
        self.files = [f for f in self.files if f.find("_SUCCESS") < 0]
        self.files.sort()

        self.is_hdfs = data_path.startswith("hdfs")
        self.repeat = repeat
        logger.info("[DATA]--all dataset containing {} files.".format(len(self.files)))

        if len(self.files) % self.world_size != 0:
            logger.info(
                "[DATA]--Whole dataset file num %s cannot split to worldsize %s " % (len(self.files), self.world_size)
            )
        self.verbose = verbose

    def generate(self, seed=42):
        """
        # TODO: 加cache，加prefetch
        在dataloader里调用，dataloader会启num_worker个进程来遍历dataset。
        这个函数一开始做了两次split_shard，对文件进行划分。
        self.files是总的数据集的文件数，
        第一次split_shard是基于rank和world_size来分的，是gpu节点维度的；
        第二次split_shard是基于worker_info的，是一个gpu节点的dataloader内，给不同的worker划分文件。

        """

        # 第一次 split: 按 rank 划分。
        # 先对files做一次sort和（seed）shuffle，每个rank拿到的seed都是一样的。这样得到的list是一样的，保证split时不重复。
        # TODO: 这里的seed实际一直一样，后面可以想办法，让seed 从trainer 传过来，让每个epoch里，每个rank拿到的file不一样，更好的做shuffle。
        if self.shuffle:
            self.files = self.sort_and_shuffle(self.files, seed)
        else:
            self.files.sort()
        if self.world_size == 1 or len(self.files) == 1:
            cur_dataloader_files = self.files
        else:
            cur_dataloader_files = split_shard(self.files, self.rank, self.world_size)

        # 第二次 split：各个rank内部，将 cur_dataloader_files 按 num_workers 分。注意每个worker都会执行。
        # 每个rank的每个 worker 拿到的都是这个：cur_dataloader_files，是一样的
        while True:
            worker_info = torch.utils.data.get_worker_info()

            if worker_info is not None:
                if len(cur_dataloader_files) % worker_info.num_workers != 0 and self.verbose:
                    logger.info(
                        "[DATA]--current dataloader [%s] file num %s cannot split to worker_num %s "
                        % (
                            self.rank,
                            len(cur_dataloader_files),
                            worker_info.num_workers,
                        )
                    )
                # 这里是真正做第二次split的地方， cur_worker_files 是每个worker 拿到的
                cur_worker_files = split_shard(
                    cur_dataloader_files,
                    worker_info.id,
                    worker_info.num_workers,
                )
            else:
                # num_worker=0，只有主进程的情况
                cur_worker_files = cur_dataloader_files

            if self.shuffle:  # 每个epoch下，虽然每个rank-每个worker对应的files是一样的，但会这里shuffle一下，读的顺序按file有打乱。
                random.shuffle(cur_worker_files)
            # cur_worker_files 是每个worker拿到的结果
            if self.verbose:
                logger.info(
                    f"[DataLoader] --> Rank:[{self.rank}]  Workers:[{worker_info.id if worker_info else 0}] "
                    "process file: {len(cur_worker_files)} :{self.get_surfix(cur_worker_files[:3])}  ..."
                )
            for filepath in cur_worker_files:
                if self.is_hdfs:
                    with hopen(filepath, "r") as reader:
                        for line in reader:
                            yield line.decode()
                    continue
                with open(filepath, "r") as reader:
                    for line in reader:
                        yield line

            if not self.repeat:
                break

    def __iter__(self):
        return self.generate()

    def reset(self, seed):
        return self.generate(seed)

    def sort_and_shuffle(self, data, seed):
        data.sort()
        random.Random(seed).shuffle(data)
        return data

    def get_surfix(self, name_list):
        return [n.split("/")[-1] for n in name_list]


def split_shard(data: List[Any], shard_idx: int, shard_size: int):
    num = len(data)
    if num < shard_size:
        raise RuntimeError("num:{} < shard size:{}".format(num, shard_size))
    start_idx = (num * shard_idx) // shard_size
    end_idx = (num * (shard_idx + 1)) // shard_size
    return data[start_idx:end_idx]
