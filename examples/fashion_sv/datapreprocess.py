import json

import numpy as np
from tqdm import tqdm

from easyguard.utils.hdfs_utils import hlist_files, hopen

idx = 0
skip = 0
# cache_dir = './cache'
cache_dir = "/mnt/bn/wxnas/hivesamples"
cache_limit = 50000
cache = []
uidcount = dict()
files = hlist_files(["hdfs://haruna/home/byte_ecom_govern/user/wangxian/datasets/fashionaudio/audio4sv/raw_0408_0409"])
files = [f for f in files if "_SUCCESS" not in f]
files = [f for f in files if int(f.split("-")[1]) < 1000]
# files = ['audio_parts']
for file in tqdm(files):
    with hopen(file, "r") as f:
        lines = f.readlines()

    for line in lines:
        try:
            if isinstance(line, str):
                dline = line
            elif isinstance(line, bytes):
                dline = str(line, encoding="utf-8")
            user_id, room_id, snapshot_id, info, *args = dline.split("\t")
        except Exception as e:
            print("error occurred:", type(line), e)
            skip += 1
            if skip % 100 == 0:
                print(f"skipped: {skip} samples")
            continue
        voice = json.loads(info)
        audio_urls = []
        for audioslice in voice["voice_text"]:
            if len(audioslice["text"]) > 25:
                audio_urls.append(audioslice["audio_url"])
        if len(audio_urls) > 2:
            sample = {
                "user_id": user_id,
                "room_id": room_id,
                "snapshot_id": snapshot_id,
                "audio_urls": audio_urls,
            }
            if user_id not in uidcount:
                uidcount[user_id] = 0
            uidcount[user_id] += len(audio_urls)
            cache.append(json.dumps(sample))

            if len(cache) >= cache_limit:
                with open(f"{cache_dir}/audio_samples_{idx}.jsonl", "w") as f:
                    f.writelines("\n".join(cache))
                cache = []
                idx += 1

if cache:
    with open(f"{cache_dir}/audio_samples_{idx}.jsonl", "w") as f:
        f.writelines("\n".join(cache))
    cache = []
    idx += 1

np.save("uidcount.npy", uidcount)
