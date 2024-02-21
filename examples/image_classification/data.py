""" Dataset."""
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == "bicubic":
            return InterpolationMode.BICUBIC
        elif method == "lanczos":
            return InterpolationMode.LANCZOS
        elif method == "hamming":
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR

    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp

from cruise.data_module import CruiseDataModule


class MyDataModule(CruiseDataModule):
    def __init__(
        self,
        data_path: str = "/mnt/bn/multimodel-pretrain/database/",
        train_split: str = "/mnt/bn/multimodel-pretrain/database/train_list.txt",
        val_split: str = "/mnt/bn/multimodel-pretrain/database/val_list.txt",
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        self.save_hparams()

    def local_rank_zero_prepare(self) -> None:
        pass

    def setup(self):
        # every process will run this after prepare is done
        self.train_dataset = DataSet(
            self.hparams.data_path,
            self.hparams.train_split,
            transform=build_transform(mode="train"),
        )
        self.val_dataset = DataSet(
            self.hparams.data_path,
            self.hparams.val_split,
            transform=build_transform(mode="val"),
        )

    def train_dataloader(self):
        sampler_train = torch.utils.data.DistributedSampler(
            self.train_dataset,
            num_replicas=int(os.environ.get("WORLD_SIZE") or 1),
            rank=int(os.environ.get("RANK") or 0),
            shuffle=True,
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        sampler_val = torch.utils.data.DistributedSampler(
            self.val_dataset,
            num_replicas=int(os.environ.get("WORLD_SIZE") or 1),
            rank=int(os.environ.get("RANK") or 0),
            shuffle=False,
        )
        return torch.utils.data.DataLoader(
            self.val_dataset,
            sampler=sampler_val,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=False,
        )


class DataSet(torch.utils.data.Dataset):
    """Common dataset."""

    def __init__(self, data_path, split, transform=None):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        self._data_path, self._split = data_path, split
        self.transform = transform
        self._construct_imdb()

    def _construct_imdb(self):
        """Constructs the imdb."""
        # Compile the split data path
        self._imdb, self._class_ids = [], []
        with open(self._split, "r") as fin:
            for line in fin:
                info = line.strip().split(" ")
                im_dir, cont_id = info[0], info[1]
                im_path = os.path.join(self._data_path, im_dir)
                self._imdb.append({"im_path": im_path, "class": int(cont_id)})
                self._class_ids.append(int(cont_id))

    def __getitem__(self, index):
        # Load the image
        try:
            im = Image.open(self._imdb[index]["im_path"])
        except:
            # index = random.randint(0, len(self._imdb))
            # im = Image.open(self._imdb[index]["im_path"])
            random_img = np.random.rand(384, 384, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        im = im.convert("RGB")
        im = self.transform(im)

        label = self._imdb[index]["class"]
        return im, label

    def __len__(self):
        return len(self._imdb)


def build_transform(mode: str = "train"):
    resize_im = True
    if mode == "train":
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5-inc1",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            interpolation="bicubic",
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(224, padding=4)
        return transform

    elif mode == "val" or mode == "test":
        t = []
        size = int((256 / 224) * 224)
        t.append(
            transforms.Resize(size, interpolation=_pil_interp("bicubic")),
            # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(224))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    return transforms.Compose(t)
