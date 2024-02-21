# -*- coding: utf-8 -*-
import base64
import io
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from cruise.data_module import CruiseDataModule
from cruise.utilities.hdfs_io import hopen
from PIL import Image
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset

from easyguard.appzoo.multimodal_modeling.utils import BertTokenizer

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

import logging

log = logging.getLogger(__name__)


class UnHqDataModule(CruiseDataModule):
    def __init__(
        self,
        train_path: str = "/mnt/bn/ecom-tianke-lq/data/unauthentic_data_live/v2/unauthentic_data_live_20230213_all/unauthentic_data_0213.csv",
        val_path: str = "/mnt/bn/ecom-tianke-lq/data/unauthentic_data_live/v1/unauthentic_data_live_process/unauthentic_data_product.csv",
        predict_path: str = "/mnt/bn/ecom-tianke-lq/data/unauthentic_data_live/v1/unauthentic_data_live_process/unauthentic_data_product.csv",
        train_frame_root: str = "/mnt/bn/ecom-tianke-lq/data/unauthentic_data_live/v2/unauthentic_data_live_20230213_all",
        val_frame_root: str = "/mnt/bn/ecom-tianke-lq/data/unauthentic_data_live/v1/unauthentic_data_live_process",
        vocab_file: str = "hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/albert_6l_zh_mix_oldcut_20200921/archer/zh_old_cut_145607.vocab",
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
        num_workers: int = 8,
        val_step: int = -1,
        # ocr_max_len: int = 40,
        product_max_len: int = 30,
        asr_max_len: int = 360,
        frame_len: int = 10,
        product_image_len: int = 1,
    ):
        super().__init__()
        self.save_hparams()

    def local_rank_zero_prepare(self) -> None:
        pass

    def setup(self):
        # every process will run this after prepare is done
        self.train_path = self.hparams.train_path
        self.val_path = self.hparams.val_path
        self.predict_path = self.hparams.predict_path

        self.params = {
            "product_max_len": self.hparams.product_max_len,
            "asr_max_len": self.hparams.asr_max_len,
            "vocab_file": self.hparams.vocab_file,
            "frame_len": self.hparams.frame_len,
            "product_image_len": self.hparams.product_image_len,
            "train_frame_root": self.hparams.train_frame_root,
            "val_frame_root": self.hparams.val_frame_root,
        }

    def train_dataloader(self):
        self.train_dataset = MMDataset(self.params, self.train_path, is_training=True)

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
            prefetch_factor=4,
            drop_last=True,
            collate_fn=self.train_dataset.collect_fn,
        )

    def val_dataloader(self):
        self.val_dataset = MMDataset(self.params, self.val_path, is_training=False)

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
            prefetch_factor=4,
            drop_last=False,
            collate_fn=self.val_dataset.collect_fn,
        )

    def predict_dataloader(self):
        self.predict_dataset = MMDataset(self.params, self.predict_path, is_training=False)

        sampler_predict = torch.utils.data.DistributedSampler(
            self.predict_dataset,
            num_replicas=int(os.environ.get("WORLD_SIZE") or 1),
            rank=int(os.environ.get("RANK") or 0),
            shuffle=False,
        )
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            sampler=sampler_predict,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.predict_dataset.collect_fn,
        )


class MMDataset(Dataset):
    """ """

    def __init__(self, params, data_path, is_training):
        super().__init__()
        self.preprocess = get_transform(mode="train" if is_training else "val")

        self.max_len = {
            "product_info": params["product_max_len"],
            "asr_text": params["asr_max_len"],
        }
        self.data_path = data_path
        self.frame_root = params["train_frame_root"] if is_training else params["val_frame_root"]
        self.frame_len = params["frame_len"]
        self.product_image_len = params["product_image_len"]
        self.tokenizer = BertTokenizer(
            params["vocab_file"],
            do_lower_case=True,
            tokenize_emoji=False,
            greedy_sharp=False,
        )

        self.PAD = self.tokenizer.vocab["[PAD]"]
        with hopen(
            "hdfs://haruna/home/byte_search_nlp_lq/multimodal/black_frame.jpg",
            "rb",
        ) as f:
            self.black_frame = self.preprocess(self._load_image(f.read()))

        self.is_training = is_training
        self.text_types = ["asr_text", "product_info"]
        # self.text_types = ['asr_text']

        self.extensions = [".jpg", ".jpeg", ".bmp", ".png"]

        self.params = params

        self.vids = []
        self.num_frames = []
        self.token_ids = []
        self.label_array = []
        self.label_key = "label_score"
        self.object_ids = []
        self.verify_results = []

        self.product_vids = []
        self.product_ids = []

        fin = pd.read_csv(self.data_path)
        for index in range(len(fin)):
            if index % 50000 == 0:
                print("read {} data".format(index), file=sys.stderr)
            object_id = "{}_{}_{}".format(
                str(fin["room_id"][index]),
                str(fin["start_time"][index]),
                str(fin["end_time"][index]),
            )
            verify_result = fin["verify_result"][index]

            video_path = os.path.join(self.frame_root, "frames_down_data", object_id)

            if not os.path.exists(video_path):
                continue

            frame_names = []
            with os.scandir(video_path) as it:
                for file in it:
                    frame_names.append(file.name)

            # frame_names = filter(lambda s: os.path.splitext(s)[-1] in self.extensions, os.listdir(video_path))
            # frame_names = sorted(frame_names, key=lambda x: int(str(x).split('.')[0]))

            if len(frame_names) <= 0:
                continue

            label = fin[self.label_key][index]
            try:
                label_score = int(label)
            except:
                continue

            frame_paths = [os.path.join(video_path, frame_name) for frame_name in frame_names]

            self.object_ids.append(object_id)
            self.verify_results.append(verify_result)

            self.vids.append(frame_paths)
            self.label_array.append(label)

            product_info_dict = eval(fin["product_info"][index])
            product_name = product_info_dict["product_name"]
            if "shop_info" in product_info_dict:
                shop_name = product_info_dict["shop_info"]["shop_name"]
                product_info = str(shop_name) + " " + str(product_name)
            else:
                product_info = str(product_name)

            # process product image
            product_id = product_info_dict["product_id"]
            product_path = os.path.join(self.frame_root, "product_down_data", str(product_id))

            if os.path.exists(product_path):
                product_image_names = os.listdir(product_path)
                product_image_paths = [
                    os.path.join(product_path, product_image_name) for product_image_name in product_image_names
                ]
                self.product_vids.append(product_image_paths)
            else:
                self.product_vids.append([])

            # product_picture_names = filter(lambda s: os.path.splitext(s)[-1] in self.extensions, os.listdir(product_path))
            # product_picture_names = sorted(product_picture_names, key=lambda x: int(str(x).split('.')[0]))

            # for product_picture in product_picture_names:
            # frame_paths.append(os.path.join(product_path, product_picture))
            # self.vids.append(frame_paths)

            asr_text = fin["asr_text"][index]

            texts = {"product_info": product_info, "asr_text": str(asr_text)}
            self.product_ids.append(product_info)
            self.token_ids.append(texts)

        if self.is_training:
            print("Train dataset init success ...", file=sys.stderr)
        else:
            print("val dataset init success ...", file=sys.stderr)

    def __getitem__(self, index):
        token_ids = self.text_preprocess(self.token_ids[index])
        product_ids = self.product_text_preprocess(self.product_ids[index])

        video_vids = self.vids[index]
        object_id = self.object_ids[index]
        verify_result = self.verify_results[index]

        product_images = []
        product_image_paths = self.product_vids[index]

        if product_image_paths == []:
            product_images = []
        for product_image_path in product_image_paths[: self.product_image_len]:
            try:
                product_image = self.image_preprocess(product_image_path)
                product_images.append(product_image)
            except:
                continue

        frames = []
        num = min(self.frame_len, len(video_vids))
        frames_raw = video_vids[:num]

        for frame_path in frames_raw:
            try:
                image_tensor = self.image_preprocess(frame_path)
                frames.append(image_tensor)
            except Exception as e:
                continue

        label = int(self.label_array[index]) - 2
        """
        label = 1 if str(self.label_array[index]) == '2' else 0
        """
        input_dict = {
            "frames": frames,
            "product_frames": product_images,
            "input_ids": token_ids,
            "product_ids": product_ids,
            "label": label,
            "object_id": object_id,
            "verify_result": verify_result,
        }
        return input_dict

    def __len__(self):
        return len(self.vids)

    def collect_fn(self, data):
        labels = []
        input_ids = []
        input_mask = []
        input_segment_ids = []
        frames = []
        frames_mask = []
        object_ids = []
        verify_results = []

        product_ids = []
        product_mask = []
        product_segment_ids = []
        product_images = []
        product_images_mask = []

        max_len = max([len(b["input_ids"]) for b in data])

        for ib, ibatch in enumerate(data):
            labels.append(ibatch["label"])
            object_ids.append(ibatch["object_id"])
            verify_results.append(ibatch["verify_result"])
            input_ids.append(ibatch["input_ids"][:max_len] + [self.PAD] * (max_len - len(ibatch["input_ids"])))

            input_mask.append([1] * len(ibatch["input_ids"][:max_len]) + [0] * (max_len - len(ibatch["input_ids"])))
            input_segment_ids.append([0] * max_len)

            product_ids.append(ibatch["product_ids"][:30] + [self.PAD] * (30 - len(ibatch["product_ids"])))
            product_mask.append([1] * len(ibatch["product_ids"][:30]) + [0] * (30 - len(ibatch["product_ids"])))
            product_segment_ids.append([0] * 30)

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

            product_frames_cur = []
            product_frames_mask_cur = []
            for img in ibatch["product_frames"]:
                product_frames_cur.append(img)
                product_frames_mask_cur.append(1)
            while len(product_frames_cur) < self.product_image_len:
                product_frames_cur.append(self.black_frame)
                product_frames_mask_cur.append(0)
            product_images.append(torch.stack(product_frames_cur, dim=0))
            product_images_mask.append(product_frames_mask_cur)

        res = {
            "product_images": torch.stack(product_images, dim=0),
            "product_images_mask": torch.tensor(product_images_mask),
            "frames": torch.stack(frames, dim=0),
            "frames_mask": torch.tensor(frames_mask),
            "input_ids": torch.tensor(input_ids),
            "input_mask": torch.tensor(input_mask),
            "input_segment_ids": torch.tensor(input_segment_ids),
            "product_ids": torch.tensor(product_ids),
            "product_mask": torch.tensor(product_mask),
            "product_segment_ids": torch.tensor(product_segment_ids),
            "labels": torch.tensor(labels, dtype=torch.long),
            "item_ids": object_ids,
            "verify_results": verify_results,
        }
        return res

    # def text_preprocess(self, texts):
    #     tokens = ['[CLS]']
    #     for text_type in self.text_types:
    #         text = texts[text_type]
    #         tokens += self.tokenizer.tokenize(text)[:self.max_len[text_type] - 2] + ['[SEP]']
    #     token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    #     return token_ids

    def text_preprocess(self, texts):
        tokens = ["[CLS]"]
        for text_type in self.text_types:
            text = texts[text_type][: self.max_len[text_type] - 2]
            tokens += self.tokenizer.tokenize(text) + ["[SEP]"]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

    def product_text_preprocess(self, text):
        tokens = ["[CLS]"]
        tokens += self.tokenizer.tokenize(text)[:29]
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
