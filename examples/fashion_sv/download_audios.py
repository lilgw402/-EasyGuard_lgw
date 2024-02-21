import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from datetime import datetime
from random import shuffle

import numpy as np
import requests

# from hdfs_utils import hlist_files, hopen


uid2label = np.load("uid2label.npy", allow_pickle=True).item()


def download_audio(url: str, timeout=2):
    response = requests.get(url, timeout=timeout)
    wav_data = response.content
    return wav_data


def process_sample(line):
    sample = json.loads(line)

    if sample["user_id"] not in uid2label:
        return None
    if "audio_urls" not in sample or len(sample["audio_urls"]) <= 1:
        return None

    if not os.path.exists(f"/mnt/bn/multimodel-pretrain/fashoin_sv/audio4sv/{sample['user_id']}"):
        try:
            os.makedirs(f"/mnt/bn/multimodel-pretrain/fashoin_sv/audio4sv/{sample['user_id']}")
        except:  # noqa: E722
            pass

    sample["audios"] = []
    for url in sample["audio_urls"]:
        try:
            wav_data = download_audio(url)
            if len(wav_data) < 500000:
                continue
            filename = url.split("/")[-1]
            assert not os.path.exists(
                f"/mnt/bn/multimodel-pretrain/fashoin_sv/audio4sv/{sample['user_id']}/{filename}"
            )
            with open(
                f"/mnt/bn/multimodel-pretrain/fashoin_sv/audio4sv/{sample['user_id']}/{sample['room_id']}-{filename}",
                "wb",
            ) as f:
                f.write(wav_data)
            sample["audios"].append(f"{sample['user_id']}/{sample['room_id']}-{filename}")
        except Exception as e:  # noqa: F841
            continue

    if len(sample["audios"]) == 0:
        return None

    return json.dumps(sample)


def download_by_file_mutlithread(file):
    with open(file, "r") as f:
        lines = f.readlines()

    shuffle(lines)
    cache = []
    with ThreadPoolExecutor(max_workers=30) as t:
        thread_group = list(t.map(process_sample, lines))
        for c in thread_group:
            cache.append(c)

    filename = file.split("/")[-1]
    cache = [c for c in cache if c is not None]

    with open(f"/mnt/bn/multimodel-pretrain/fashoin_sv/data_jsonl/{filename}", "w") as f:
        f.writelines("\n".join(cache))

    print(f"{datetime.now()} finished: {filename} -- {len(cache)}/{len(lines)}")


# def download_by_file(file):
#     with hopen(file, 'r') as f:
#         lines = f.readlines()

#     cache = []
#     for line in tqdm(lines[:30]):
#         r = process_sample(line)
#         if r is not None:
#             cache.append(r)

#     filename = file.split('/')[-1]
#     cache = [c for c in cache if c is not None]

#     with open(f'/mnt/bd/wx-nas/dataset/data_jsonls/{filename}', 'w') as f:
#         f.writelines('\n'.join(cache))

#     print(f'{datetime.now()} finished: {filename} -- {len(cache)}/{len(lines)}')


# def get_files(p):
#     if isinstance(p, str):
#         p = [p]
#     files = hlist_files(p)
#     files = [f for f in files if f.find('_SUCCESS') < 0]
#     return files


if __name__ == "__main__":
    # files = get_files('/mnt/bd/wx-nas/audiosamples/*.jsonl')
    from glob import glob

    files = glob("./audiosamples/*.jsonl")
    files = [f for f in files if int(f.split("_")[-1].split(".")[0]) < 66]
    print(len(files))
    start = time.time()
    with ProcessPoolExecutor(max_workers=10) as p:
        futures = [
            p.submit(download_by_file_mutlithread, file)
            for file in files
            if not os.path.exists(f"/mnt/bn/multimodel-pretrain/fashoin_sv/data_jsonl/{file.split('/')[-1]}")
        ]
        wait(futures)
    # download_by_file(files[0])
    print(f"finished: {time.time() - start}")
