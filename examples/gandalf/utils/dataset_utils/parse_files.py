import os
import random

from utils.driver import TRIAL_ID
from utils.file_util import safe_int, scan_hdfs_dir, scan_local_dir

data_type_map = {
    "ParquetDataFactory": "parquet",
    "TFRecordDataFactory": "tfrecord",
    "JsonLDataFactory": "jsonl",
    "KVDataFactory": "kv",
}


def get_ds_path(
    folder_path_str,
    folder_str,
    data_type,
    fname_pattern,
    fmin_size=None,
    group_keys=None,
    shuffle=False,
):
    if not fmin_size:
        fmin_size = None
    data_type = data_type_map[data_type]
    if not fname_pattern:
        if data_type == "jsonl":
            fname_pattern = ""
        elif data_type == "kv":
            fname_pattern = "*index"
        else:
            fname_pattern = "*{}".format(data_type)
    folders = folder_str.split("$")
    if "$" in folder_path_str:
        folder_paths = folder_path_str.split("$")
    else:
        folder_paths = [folder_path_str] * len(folders)
    data_sources, data_types = [], []
    for folder_path, folder in zip(folder_paths, folders):
        if data_type == "tfrecord":
            # for tfrecord, follow the logic in tfrecord data factory
            full_path = os.path.join(folder_path, folder, fname_pattern)
            folder_path, folder, fname_pattern = full_path.rsplit("/", 2)
        if folder_path.startswith("hdfs"):
            cur_data_source = scan_hdfs_dir(
                folder_path,
                folder,
                fname_pattern,
                stick_folder_file=data_type == "kv",
                min_size=fmin_size,
            )
        else:
            cur_data_source = []
            for cur_folder in folder.split("|"):
                cur_data_source += scan_local_dir(folder_path, cur_folder, fname_pattern)
        # remove duplicate files
        cur_data_type = data_type
        cur_data_source = list(set(cur_data_source))
        cur_data_source.sort()
        if shuffle:
            random.Random(safe_int(TRIAL_ID or 42)).shuffle(cur_data_source)
        if data_type == "kv":
            cur_data_source = [x[:-6] for x in cur_data_source]
            if group_keys:
                for key in group_keys:
                    _tmp = [x for x in cur_data_source if key in x]
                    data_sources.append(_tmp)
                    data_types.append(cur_data_type)
        else:
            data_sources.append(cur_data_source)
            data_types.append(cur_data_type)
    return data_sources, data_types
