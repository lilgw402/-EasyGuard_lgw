# -*- coding: utf-8 -*-
import base64
import io
import json
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from cruise.data_module import CruiseDataModule
from cruise.utilities.hdfs_io import hopen
from PIL import Image
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset

from easyguard.appzoo.multimodal_modeling.utils import BertTokenizer
from easyguard.utils.data_helpers import build_vocab

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
except:  # noqa: E722
    from timm.data.transforms import _pil_interp

import logging

log = logging.getLogger(__name__)


class LiveSiteDataModule(CruiseDataModule):
    def __init__(
        self,
        train_path: str = "/mnt/bn/suyulin/benchmark/live_site_label_data/site_new_2/train_metas.json",
        val_path: str = "/mnt/bn/suyulin/benchmark/live_site_label_data/site_new_2/val_metas.json",
        frame_root: str = "/mnt/bn/suyulin/benchmark/live_site_label_data/site_new_2/local_frames",
        vocab_file: str = "hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/albert_6l_zh_mix_oldcut_20200921/archer/zh_old_cut_145607.vocab",  # noqa: E501
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 8,
        val_step: int = -1,
        ocr_max_len: int = 40,
        asr_max_len: int = 360,
        title_max_len: int = 30,
        frame_len: int = 12,
        num_classes_lv1: int = 4,
        num_classes_lv2: int = 16,
        augment: str = "",
    ):
        super().__init__()
        self.save_hparams()

    def local_rank_zero_prepare(self) -> None:
        pass

    def setup(self):
        # every process will run this after prepare is done
        self.train_path = self.hparams.train_path
        self.val_path = self.hparams.val_path

        self.params = {
            "ocr_max_len": self.hparams.ocr_max_len,
            "asr_max_len": self.hparams.asr_max_len,
            "title_max_len": self.hparams.title_max_len,
            "vocab_file": self.hparams.vocab_file,
            "frame_len": self.hparams.frame_len,
            "frame_root": self.hparams.frame_root,
            "augment": self.hparams.augment,
        }

        self.train_dataset = MMDataset(self.params, self.train_path, is_training=True)

        self.val_dataset = MMDataset(self.params, self.val_path, is_training=False)

        self.test_dataset = MMDataset(self.params, self.val_path, is_training=False)

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
            collate_fn=self.train_dataset.collect_fn,
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
            collate_fn=self.val_dataset.collect_fn,
        )

    def test_dataloader(self):
        sampler_test = torch.utils.data.DistributedSampler(
            self.test_dataset,
            num_replicas=int(os.environ.get("WORLD_SIZE") or 1),
            rank=int(os.environ.get("RANK") or 0),
            shuffle=False,
        )
        return torch.utils.data.DataLoader(
            self.test_dataset,
            sampler=sampler_test,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.test_dataset.collect_fn,
        )

    def cal_num_per_cls(self):
        num_per_cls_lv1 = [0 for i in range(self.hparams.num_classes_lv1)]
        num_per_cls_lv2 = [0 for i in range(self.hparams.num_classes_lv2)]
        # load data
        fin = open(self.hparams.train_path)
        for sample in fin.readlines():
            data_item = json.loads(sample)
            if data_item["num_frame"] == 0:
                continue
            label1 = data_item["lv1_label"]
            label2 = data_item["lv2_label"]
            num_per_cls_lv1[int(label1)] += 1
            num_per_cls_lv2[int(label2)] += 1

        return num_per_cls_lv1, num_per_cls_lv2


class MMDataset(Dataset):
    """ """

    def __init__(self, params, data_path, is_training=False):
        super().__init__()
        self.preprocess = get_transform(mode="train" if is_training else "val")

        self.max_len = {
            "text_ocr": params["ocr_max_len"],
            "text_asr": params["asr_max_len"],
            "room_title": params["title_max_len"],
        }
        self.data_path = data_path
        self.frame_root = params["frame_root"]
        self.frame_len = params["frame_len"]
        self.tokenizer = BertTokenizer(
            params["vocab_file"],
            do_lower_case=True,
            tokenize_emoji=False,
            greedy_sharp=False,
        )

        self.PAD = build_vocab(params["vocab_file"])["[PAD]"]
        with hopen(
            "hdfs://haruna/home/byte_search_nlp_lq/multimodal/black_frame.jpg",
            "rb",
        ) as f:
            self.black_frame = self.preprocess(self._load_image(f.read()))

        self.training = is_training
        self.text_types = ["text_asr", "text_ocr", "room_title"]

        self.params = params

        # load data
        fin = open(self.data_path)

        self.vids = []
        self.num_frames = []
        self.token_ids = []
        self.label_array1, self.label_array2 = [], []
        self.label_key = ["lv1_label", "lv2_label"]

        for sample in fin.readlines():
            data_item = json.loads(sample)
            if data_item["num_frame"] == 0:
                continue
            self.vids.append(data_item["vid"])
            self.num_frames.append(int(data_item["num_frame"]))

            label1, label2 = (
                data_item[self.label_key[0]],
                data_item[self.label_key[1]],
            )
            self.label_array1.append(label1)
            self.label_array2.append(label2)

            texts = {
                "text_ocr": data_item["ocr_text"],
                "text_asr": data_item["asr_text"],
                "room_title": data_item["room_title"],
            }
            self.token_ids.append(texts)
        fin.close()

    def __getitem__(self, index):
        token_ids = self.text_preprocess(self.token_ids[index])
        video_vid = self.vids[index]

        frames = []
        num = min(self.frame_len, self.num_frames[index])
        frames_raw = [video_vid + f"/{fno}.png" for fno in range(self.num_frames[index])][:num]

        for frame_path in frames_raw:
            image_tensor = self.image_preprocess(os.path.join(self.frame_root, frame_path))
            frames.append(image_tensor)

        input_dict = {
            "vid": self.vids[index],
            "frames": frames,
            "input_ids": token_ids,
            "label_1": int(self.label_array1[index]),
            "label_2": int(self.label_array2[index]),
        }
        return input_dict

    def __len__(self):
        return len(self.vids)

    def collect_fn(self, data):
        vids = []
        labels_1 = []
        labels_2 = []
        input_ids = []
        input_mask = []
        input_segment_ids = []
        frames = []
        frames_mask = []

        max_len = max([len(b["input_ids"]) for b in data])
        if self.training and self.params.get("augment", "") == "mixgen":
            data = mixgen(data)

        for ib, ibatch in enumerate(data):
            vids.append(ibatch["vid"])
            labels_1.append(ibatch["label_1"])
            labels_2.append(ibatch["label_2"])

            input_ids.append(ibatch["input_ids"][:max_len] + [self.PAD] * (max_len - len(ibatch["input_ids"])))
            input_mask.append([1] * len(ibatch["input_ids"][:max_len]) + [0] * (max_len - len(ibatch["input_ids"])))
            input_segment_ids.append([0] * max_len)

            frames_cur = []
            frames_mask_cur = []
            for img in ibatch["frames"]:
                frames_cur.append(img)
                frames_mask_cur.append(1)
            while len(frames_cur) < self.frame_len:
                frames_cur.append(self.black_frame)
                frames_mask_cur.append(0)
            frames.append(torch.stack(frames_cur, dim=0))
            frames_mask.append(frames_mask_cur)

        res = {
            "frames": torch.stack(frames, dim=0),
            "frames_mask": torch.tensor(frames_mask),
            "vids": vids,
            "input_ids": torch.tensor(input_ids),
            "input_mask": torch.tensor(input_mask),
            "input_segment_ids": torch.tensor(input_segment_ids),
            "labels_1": torch.tensor(labels_1, dtype=torch.long),
            "labels_2": torch.tensor(labels_2, dtype=torch.long),
        }
        return res

    def text_preprocess(self, texts):
        tokens = ["[CLS]"]
        for text_type in self.text_types:
            text = texts[text_type][: self.max_len[text_type] - 2]
            tokens += self.tokenizer.tokenize(text) + ["[SEP]"]
            # tokens += self.tokenizer.tokenize(text[:self.max_len[text_type]])[:self.max_len[text_type] - 2] + ['[SEP]']        # noqa: E501
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

    def image_preprocess(self, image_path):
        image = Image.open(image_path)
        image_tensor = self.preprocess(image)
        return image_tensor

    @staticmethod
    def b64_decode(string):
        if isinstance(string, str):
            string = string.encode()
        return base64.decodebytes(string)

    @staticmethod
    def _load_image(buffer):
        img = Image.open(io.BytesIO(buffer)).convert("RGB")
        return img


def get_transform(mode: str = "train"):
    """
    根据不同的data，返回不同的transform
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if mode == "train":
        com_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif mode == "val":
        com_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise ValueError("mode [%s] is not in [train, val]" % mode)
    return com_transforms


def get_transform_beta(mode: str = "train"):
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

    elif mode == "val":
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


def mixgen(data, lam=0.5):
    batch_size = len(data) // 4
    index = np.random.permutation(batch_size)
    for i in range(batch_size):
        # image mixup
        for j in range(len(data[i]["frames"])):
            data[i]["frames"][j] = lam * data[i]["frames"][j] + (1 - lam) * data[index[i]]["frames"][j]
        # text concat
        data[i]["input_ids"] = data[i]["input_ids"] + data[index[i]]["input_ids"]
    return data
