import json
import os
import random

import numpy as np
import torch

# import soundfile
import torchaudio
from cruise.data_module import CruiseDataModule

from examples.fashionproduct_xl.fashoinproduct_xl_cat.dist_dataset import DistLineReadingDataset


class TorchvisionLabelDataset(DistLineReadingDataset):
    """
    dataset，继承的dataset是 DistLineReadingDataset，是一个逐行读hdfs数据的IterableDataset。
    """

    def __init__(
        self,
        config,
        data_path,
        rank=0,
        world_size=1,
        shuffle=True,
        repeat=False,
        is_training=False,
    ):
        super().__init__(data_path, rank, world_size, shuffle, repeat)
        self.config = config
        self.world_size = world_size
        self.is_training = is_training
        self.root_path = self.config.root_path
        self.num_frames = self.config.num_frames
        self.sampling_rate = self.config.sampling_rate
        self.uid2label = np.load("./examples/fashion_sv/uid2label.npy", allow_pickle=True).item()

    def __len__(self):
        # world_size = os.environ.get('WORLD_SIZE') if os.environ.get('WORLD_SIZE') is not None else 1
        if self.is_training:
            return self.config.train_size // self.world_size
        else:
            return self.config.val_size // self.world_size

    def __iter__(self):
        for example in self.generate():
            try:
                data_item = json.loads(example)

                # label
                user_id = data_item["user_id"]
                label = int(self.uid2label[user_id])

                # audio
                files = data_item["audios"]
                file = random.choice(files)
                filepath = f"{self.root_path}/{file}"
                # audio, sr = soundfile.read(filepath)
                audio, sr = torchaudio.load(filepath)
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

                yield input_dict

            except Exception as e:
                print(f"error in dataset: {e}")
                pass

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
        self.train_dataset = TorchvisionLabelDataset(
            self.hparams,
            data_path=self.hparams.train_files,
            rank=int(os.environ.get("RANK") or 0),
            world_size=int(os.environ.get("WORLD_SIZE") or 1),
            shuffle=True,
            repeat=True,
            is_training=True,
        )
        print(f"len of trainset: {len(self.train_dataset)}")

        self.val_dataset = TorchvisionLabelDataset(
            self.hparams,
            data_path=self.hparams.val_files,
            rank=int(os.environ.get("RANK") or 0),
            world_size=int(os.environ.get("WORLD_SIZE") or 1),
            # world_size=1,
            shuffle=False,
            repeat=False,
            is_training=False,
        )

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.train_dataset.collect_fn,
        )
        print(len(train_loader))
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.train_dataset.collect_fn,
        )
        return val_loader
