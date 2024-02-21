import json
import os
import random
import shutil
import time
import uuid

import tqdm
from dataloader import KVWriter, merge

from examples.utils.multi_process import run_processes_args_list


def aggregate_all_ocr_texts(ocr_lst):
    res = []
    for sub_lst in ocr_lst:
        for obj in sub_lst:
            cur_text = obj["text"]
            res.append(cur_text)
    return "".join(res)


def get_asr_ocr(asr_dir, item_id):
    data_path = os.path.join(asr_dir, f"{item_id}.json")
    raw_data = json.load(open(data_path, "r", encoding="utf-8"))
    try:
        ocr_text = json.loads(raw_data["ocr_text_all_str"])
        ocr_text = aggregate_all_ocr_texts(ocr_text)
    except:  # noqa: E722
        ocr_text = ""
    asr_text = raw_data["voice_text"]
    verify_score = json.loads(raw_data["verify_score"])[0]
    label = -1
    if verify_score == "2分":
        label = 0
    elif verify_score == "3分":
        label = 1
    elif verify_score == "4分" or verify_score == "5分":
        label = 2
    else:
        raise ValueError("invalid verify_score: {}".format(verify_score))
    return asr_text, ocr_text, label


def get_video_frame_paths(video_dir, item_id):
    data_path = os.path.join(video_dir, item_id)
    file_list = os.listdir(data_path)
    # sorted in chronological order
    file_list = sorted(file_list, key=lambda x: int(x.split(".")[0]))
    file_list = [os.path.join(data_path, f) for f in file_list]
    return file_list


def pack_data(item_ids, src_root_dir, dst_root_dir, chunk_size=32):
    tcs_dir = os.path.join(src_root_dir, "tcs_meta")
    video_dir = os.path.join(src_root_dir, "modal_data/video_frames")

    idx = "tmp_" + uuid.uuid4().hex
    save_dir = f"{dst_root_dir}/{idx}/"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    num_shard = 4
    kv_writer = KVWriter(save_dir, num_shard)

    pid = os.getpid()
    tbar = tqdm.tqdm(item_ids)
    example_keys, examples = [], []
    success_cnt, fail_cnt = 0, 0
    for item_id in tbar:
        try:
            st = time.time()
            asr_text, ocr_text, label = get_asr_ocr(tcs_dir, item_id)
            frame_file_paths = get_video_frame_paths(video_dir, item_id)
            example = json.dumps(
                {
                    "asr_text": asr_text,
                    "ocr_text": ocr_text,
                    "frame_file_paths": frame_file_paths,
                    "label": label,
                }
            ).encode("utf-8")
            # print(example)
            cost_time = time.time() - st
        except Exception as ec:  # noqa: F841
            fail_cnt += 1
            continue
            # raise Exception

        success_cnt += 1
        example_keys.append(item_id)
        examples.append(example)

        description = f"{pid} {idx}: Success: {success_cnt}, Fail: {fail_cnt} Serialized Cost: {cost_time}"
        if len(examples) == chunk_size:
            st = time.time()
            kv_writer.write_many(example_keys, examples)
            cost_time = time.time() - st

            description += f"KVWrite Cost: {cost_time}"

            example_keys = []
            examples = []

        tbar.set_description(description)

    if len(example_keys) > 0:
        kv_writer.write_many(example_keys, examples)

    print(f"{pid}: Write Done")
    kv_writer.flush()


def merge_dataset(data_dir, name="merge", version="v1_0"):
    data_dir_list = os.listdir(data_dir)
    data_dir_list = [os.path.join(data_dir, f"{x}/") for x in data_dir_list if x[:3] == "tmp"]

    save_dir = f"{data_dir}/{version}/{name}"
    if os.path.exists(save_dir + ".index"):
        raise FileExistsError
    print(f"Save Merged data to: {save_dir}")
    merge(data_dir_list, save_dir)

    # if not to_hdfs:
    #     print(f"\nUpload to hdfs: {hdfs_dir}")
    #     hput(save_dir, hdfs_dir)


def rm_tmp(dst_root_dir):
    tmp_dir_list = os.listdir(dst_root_dir)
    tmp_dir_list = [os.path.join(dst_root_dir, f"{x}/") for x in tmp_dir_list if x[:3] == "tmp"]
    for tmp_dir in tmp_dir_list:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    src_root_dir = "/mnt/bn/ecom-govern-maxiangqian/liuzeyu/hq_live/datasets/20221001_20221102/"
    dst_root_dir = "/mnt/bn/ecom-govern-maxiangqian/liuzeyu/hq_live/packed_data/20221001_20221102"
    # item_lst = open('itemids_128.txt').read().splitlines()
    item_lst = (
        open("/mnt/bn/ecom-govern-maxiangqian/liuzeyu/hq_live/datasets/20221001_20221102/item_ids.txt")
        .read()
        .splitlines()
    )
    num_workers = 16
    chunk_size = 32
    version = "v1_0"
    prefix = "train_86w"
    split_trainval = True

    if not os.path.isdir(dst_root_dir):
        os.makedirs(dst_root_dir)

    # -------------------------------Split Train/Val Set---------------------------------
    if split_trainval:
        val_ratio = 0.02
        random.shuffle(item_lst)
        train_item_lst = item_lst[: int(len(item_lst) * (1 - val_ratio))]
        val_item_lst = item_lst[int(len(item_lst) * (1 - val_ratio)) :]
    # -----------------------------------------------------------------------------------

    rm_tmp(dst_root_dir)
    # pack_data(item_lst, src_root_dir, dst_root_dir, 32)

    if split_trainval:
        run_processes_args_list(
            pack_data,
            train_item_lst,
            num_workers,
            src_root_dir=src_root_dir,
            dst_root_dir=dst_root_dir,
            chunk_size=chunk_size,
        )
        print("Pack Train Dataset Done.")
        print(f"Merge into {dst_root_dir}...")
        merge_dataset(dst_root_dir, prefix + "_train", version)

        rm_tmp(dst_root_dir)

        run_processes_args_list(
            pack_data,
            val_item_lst,
            num_workers,
            src_root_dir=src_root_dir,
            dst_root_dir=dst_root_dir,
            chunk_size=chunk_size,
        )
        print("Pack Val Dataset Done.")
        print(f"Merge into {dst_root_dir}...")
        merge_dataset(dst_root_dir, prefix + "_val", version)
    else:
        run_processes_args_list(
            pack_data,
            item_lst,
            num_workers,
            src_root_dir=src_root_dir,
            dst_root_dir=dst_root_dir,
            chunk_size=chunk_size,
        )
        print("Pack Dataset Done.")
        print(f"Merge into {dst_root_dir}...")
        merge_dataset(dst_root_dir, prefix, version)
