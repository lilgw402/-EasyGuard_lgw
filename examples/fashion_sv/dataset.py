import os
import random

import numpy as np
import torch
import torchaudio
from cruise.data_module import CruiseDataModule
from torch.utils.data import Dataset, DistributedSampler


class SVDataset(Dataset):
    def __init__(self, config, data_path, is_training=False):
        super().__init__()
        self.config = config
        self.is_training = is_training
        self.root_path = self.config.root_path
        self.num_frames = self.config.num_frames
        self.sampling_rate = self.config.sampling_rate
        self.uid2label = np.load("./examples/fashion_sv/uid2label.npy", allow_pickle=True).item()
        with open(data_path, "r") as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index].strip()

        # label
        user_id = data_item.split("/")[-2]
        label = int(self.uid2label[user_id])

        # audio
        audio, sr = torchaudio.load(data_item)
        if sr != self.sampling_rate:
            print(f"source sr: {sr}")
            audio = torchaudio.transforms.Resample(sr, self.sampling_rate)(audio)
        audio = audio.squeeze(0).numpy()

        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            padding_length = length - audio.shape[0]
            audio = np.pad(audio, (0, padding_length), "wrap")
        start_frame = int(random.random() * (audio.shape[0] - length))  # 截断
        audio = audio[start_frame : start_frame + length]
        audio = torch.from_numpy(audio).float()

        # todo Data Augmentation - add noise

        input_dict = {"audio": audio, "label": label}

        return input_dict

    def collect_fn(self, data):
        feature = []
        labels = []

        for ib, ibatch in enumerate(data):
            feature.append(ibatch["audio"])
            labels.append(ibatch["label"])

        feature = torch.stack(feature, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        res = {"feature": feature, "labels": labels}

        return res


class SVDataModule(CruiseDataModule):
    def __init__(
        self,
        root_path: str = None,
        train_files: str = None,
        train_size: int = 1500000,
        val_files: str = None,
        val_size: int = 16000,
        train_batch_size: int = 64,
        val_batch_size: int = 32,
        num_frames: int = 200,
        sampling_rate: int = 16000,
        num_workers: int = 8,
        exp: str = "default",
        download_files: list = [],
    ):
        super().__init__()
        self.save_hparams()

    def local_rank_zero_prepare(self) -> None:
        # download cutter resource
        if self.hparams.download_files:
            to_download = [df.split("->") for df in self.hparams.download_files]
            for src, tar in to_download:
                if not os.path.exists(tar):
                    os.makedirs(tar)
                fdname = src.split("/")[-1]
                if os.path.exists(f"{tar}/{fdname}"):
                    print(f"{tar}/{fdname} already existed, pass!")
                else:
                    print(f"downloading {src} to {tar}")
                    os.system(f"hdfs dfs -get {src} {tar}")

    def setup(self, stage) -> None:
        self.train_dataset = SVDataset(self.hparams, data_path=self.hparams.train_files, is_training=True)

        self.val_dataset = SVDataset(self.hparams, data_path=self.hparams.val_files, is_training=False)

    def train_dataloader(self):
        sampler_train = DistributedSampler(
            self.train_dataset,
            num_replicas=int(os.environ.get("WORLD_SIZE") or 1),
            rank=int(os.environ.get("RANK") or 0),
            shuffle=True,
        )

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.train_dataset.collect_fn,
        )
        return train_loader

    def val_dataloader(self):
        sampler_val = DistributedSampler(
            self.val_dataset,
            num_replicas=int(os.environ.get("WORLD_SIZE") or 1),
            rank=int(os.environ.get("RANK") or 0),
            shuffle=False,
        )

        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            sampler=sampler_val,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.val_dataset.collect_fn,
        )
        return val_loader
