# -*- coding: utf-8 -*-
import json
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from ecapatdnn import ECAPA_TDNN
from tqdm import tqdm

from easyguard.utils.auxiliary_utils import load_pretrained_model_weights


def processor(filepath=None, sampling_rate=16000, num_frames=None):
    audio, sr = torchaudio.load(filepath)

    if sr != sampling_rate:
        print(f"source sr: {sr}")
        audio = torchaudio.transforms.Resample(sr, sampling_rate)(audio)
    if num_frames is not None:
        audio = audio.squeeze(0).numpy()
        length = num_frames * 160 + 240
        if audio.shape[0] <= length:
            padding_length = length - audio.shape[0]
            audio = np.pad(audio, (0, padding_length), "wrap")
        start_frame = int(random.random() * (audio.shape[0] - length))  # 截断
        audio = audio[start_frame : start_frame + length]
        audio = torch.from_numpy(audio).float()
    return audio


def cos_sim(e1, e2):
    e1 = F.normalize(e1, dim=1)
    e2 = F.normalize(e2, dim=1)
    score = torch.mm(e1, e2.t())
    return score


def load_model(pretrain=None):
    svenocder = ECAPA_TDNN(C=1024, hidden_dim=192)
    if pretrain:
        renamed_ckpt = OrderedDict()
        ckpt = torch.load(pretrain, map_location="cpu")
        for n, p in ckpt["state_dict"].items():
            renamed_ckpt[n.replace("speaker_encoder.", "")] = p

        svenocder.load_state_dict(renamed_ckpt, strict=True)
    return svenocder


if __name__ == "__main__":
    # pretrain = 'hdfs://haruna/home/byte_ecom_govern/user/wangxian/projects/fashionaudio/fashoin_sv/version_13299379/
    # checkpoints/epoch=3-step=2188.ckpt'
    # pretrain = './sv_2188.ckpt'
    pretrain = None
    model = load_model(pretrain=pretrain)
    load_pretrained_model_weights(
        model,
        # 'hdfs://haruna/home/byte_ecom_govern/user/wangxian/projects/fashionaudio/
        # fashoin_sv/version_13299379/checkpoints/epoch=3-step=2188.ckpt',
        "./sv_2188.ckpt",
        rm_prefix="speaker_encoder.",
    )
    model.eval()
    path_root = "/mnt/bn/multimodel-pretrain/fashoin_sv/test_audio"
    # path1 = 'stream-112815505592811999-20230410T084910-20230410T084930.wav'
    # path2 = 'stream-112802144152191455-20230408T063130-20230408T063151.wav'
    # wav1 = processor(f'{path_root}/{path1}')
    # wav2 = processor(f'{path_root}/{path2}')
    # embed1 = model.forward(wav1, aug=False)
    # embed2 = model.forward(wav2, aug=False)
    # print(cos_sim(embed1, embed2))

    with open("./downloaded_lasted_test.jsonl", "r") as f:
        lines = f.readlines()

    acc = 0
    res = []
    for line in tqdm(lines):
        s = json.loads(line)
        s1, s2 = s["s1"], s["s2"]
        label = int(s["label"])
        wav1 = processor(f"{path_root}/{s1[0]}")
        wav2 = processor(f"{path_root}/{s2[0]}")
        embed1 = model.forward(wav1, aug=False)
        embed2 = model.forward(wav2, aug=False)
        score = cos_sim(embed1, embed2)
        s["pred"] = str(score.detach().cpu().numpy()[0][0])
        if int(score.detach().cpu().numpy()[0][0] > 0.7) == label:
            acc += 1
        else:
            print(score)
            for k, v in s.items():
                print(k, v)
        res.append(json.dumps(s))

    print(acc / len(lines))

    with open("pred_sv_score.jsonl", "w") as f:
        f.writelines("\n".join(res))

# python3 examples/fashion_sv/infer.py
