# -*- coding: utf-8 -*-

import io
import json
import os
from typing import List, Union

import numpy as np
import torch
from PIL import Image

try:
    import cruise
except ImportError:
    print(
        "[ERROR] cruise is not installed! Please refer this doc: https://bytedance.feishu.cn/wiki/wikcnGP7yzZAuKpPfL6jRJKl2ag"
    )

from cruise.data_module import CruiseDataModule
from cruise.data_module.cruise_loader import DistributedCruiseDataLoader
from cruise.utilities.hdfs_io import hexists, hlist_files, hopen

from .downloads import download_url_with_exception, get_real_url
from .utils import BertTokenizer, ImageProcess, TextProcess, load_vocab


class ImageTextProcessor:
    def __init__(
        self,
        mode,
        vocab_path="hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/zh_old_cut_145607.vocab",
        max_len=256,
        category_path="hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/category_dict_pt.json",
    ):
        self.image_process = ImageProcess(mode)
        self.text_process = TextProcess(vocab_file=vocab_path, max_len=max_len)

        """
        Load category dict for category prediction
        """
        if not hexists(category_path):
            raise ValueError("Category dict {} does not exist!".format(category_path))
        with hopen(category_path, "r") as fp:
            self.category_dict = json.load(fp)

    def transform(self, data_dict: dict):
        image_urls = data_dict["main_images"]

        product_name = data_dict["product_name"]
        image_ocr = data_dict["main_ocr"]
        if isinstance(image_ocr, list):
            image_ocr = " ".join(image_ocr)

        token_ids, text_masks, text_segment_ids = self.text_process(product_name, image_ocr)
        image = self.image_process(image_urls)

        level1_cid = str(data_dict["first_cid_new"])
        level2_cid = str(data_dict["second_cid_new"])
        label = 0
        label_l1 = 0
        if level1_cid in self.category_dict["level1"]["id2idx"]:
            label_l1 = self.category_dict["level1"]["id2idx"][level1_cid] + 1
        if level2_cid in self.category_dict["level2"]["id2idx"]:
            label = self.category_dict["level2"]["id2idx"][level2_cid] + 1

        return {
            "token_ids": token_ids,
            "image": image,
            "label": label,
            "label_l1": label_l1,
        }

    def batch_transform(self, data):
        keys = list(data[0].keys())
        batch = {k: [] for k in keys}

        for i, ibatch in enumerate(data):
            for k in keys:
                batch[k].append(ibatch[k])

        for k in keys:
            if isinstance(batch[k][0], torch.Tensor):
                batch[k] = torch.stack(batch[k], dim=0)
            elif isinstance(batch[k][0], np.ndarray):
                batch[k] = torch.from_numpy(np.stack(batch[k], axis=0))
            else:
                batch[k] = torch.tensor(batch[k])

        return batch


class MMDataModule(CruiseDataModule):
    def __init__(
        self,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        paths: Union[
            str, List[str]
        ] = "hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/pretrain_20220802_20220808_train_url",
        data_size: int = 140000000,
        val_step: int = 20,
        num_workers: int = 24,
        max_len: int = 256,
    ):
        super().__init__()
        self.save_hparams()

    def local_rank_zero_prepare(self) -> None:
        pass

    def setup(self, stage) -> None:
        paths = self.hparams.paths
        if isinstance(paths, str):
            paths = [paths]
        files = hlist_files(paths)
        files = [f for f in files if f.find("_SUCCESS") < 0]
        if not files:
            raise RuntimeError(f"No valid files can be found matching `paths`: {paths}")
        files = sorted(files)
        self.train_files = files
        self.val_files = files

    def train_dataloader(self):
        return DistributedCruiseDataLoader(
            data_sources=[self.train_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.train_batch_size],
            num_workers=self.hparams.num_workers,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=ImageTextProcessor("train"),
            predefined_steps=self.hparams.data_size // self.hparams.train_batch_size // self.trainer.world_size,
            source_types=["jsonl"],
            shuffle=True,
        )

    def val_dataloader(self):
        return DistributedCruiseDataLoader(
            data_sources=[self.val_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.val_batch_size],
            num_workers=self.hparams.num_workers // 2,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=ImageTextProcessor("val"),
            predefined_steps=self.hparams.val_step,
            source_types=["jsonl"],
            shuffle=False,
        )


"""
Text/Image Processor for FashionProduct
"""


class FPImageTextProcessor:
    def __init__(
        self,
        mode,
        vocab_path="/opt/tiger/liuyuhang/zh_deberta_base_l6_emd_20210720/vocab.txt",
        cutter_enable=False,
        cutter="/opt/tiger/liuyuhang/libcut_data_zh_20200827fix2",
        max_len=256,
        category_path="hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/category_dict_pt.json",
        ner_task_dict="hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/ner_format_statistics/ner_task_dict.json",
        ner_tasks=["商品", "颜色", "风格", "材质", "样式"],
        max_main=3,
        max_desc=5,
        max_sku=3,
    ):
        """
        Prepare text tokenizer
        """
        self.vocab = load_vocab(vocab_path)  # 载入词典
        self.max_len = max_len
        self.CLS = self.vocab["[CLS]"]  # Special token index
        self.PAD = self.vocab["[PAD]"]
        self.SEP = self.vocab["[SEP]"]
        self.MASK = self.vocab["[MASK]"]

        self.cutter_enable = cutter_enable
        if cutter_enable:
            self.cutter = Cutter("CRF_LARGE", cutter)
        # self.tokenizer = BpeTokenizer(vocab_file,
        #                               wordpiece_type="bert",
        #                               lower_case=self.config.get("text.do_lower_case", False))
        self.tokenizer = BertTokenizer(
            vocab_path,
            do_lower_case=False,
            tokenize_emoji=False,
            greedy_sharp=True,
        )

        """
        Load category dict for pretraining.
        """
        if not hexists(category_path):
            raise ValueError("Category dict {} does not exist!".format(category_path))
        with hopen(category_path, "r") as fp:
            self.category_dict = json.load(fp)

        """
        Load NER dict for pretraining
        """
        self.ner_task_dict = None
        self.ner_tasks = ner_tasks
        if len(ner_tasks) > 0:
            print("[Pretrain Tasks]--using ner tasks {} for pretraining".format(ner_tasks))
            assert hexists(ner_task_dict), "ner task dict {} does not exist!".format(ner_task_dict)
            with hopen(ner_task_dict, "r") as fp:
                self.ner_task_dict = json.load(fp)
            for task in ner_tasks:
                assert task in self.ner_task_dict, "task {} is not supported in ner tasks.".format(task)

        self.img_transform = ImageProcess(mode).transform
        self.max_main = max_main
        self.max_desc = max_desc
        self.max_sku = max_sku

    @property
    def empty_image(self):
        return b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xdb\x00C\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xc0\x00\x11\x08\x00\x02\x00\x02\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x15\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x11\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xbf\x80\x01\xff\xd9'

    def prepare_single_image(self, image, is_url=True):
        is_valid = 0
        try:
            if is_url:
                if image == "":  # Padding image
                    image_str = self.empty_image
                else:  # Valid image
                    suffix = image.split("/")[-1].split("~")[0]
                    url = get_real_url(suffix)
                    image_str = download_url_with_exception(url)

                    if image_str == b"" or image_str == "":
                        image_str = self.empty_image
                    else:
                        is_valid = 1
                image = Image.open(io.BytesIO(image_str)).convert("RGB")
            else:
                image_str = b64decode(image)
                image = Image.open(io.BytesIO(image_str)).convert("RGB")
                is_valid = 1
        except:
            image_str = self.empty_image
            image = Image.open(io.BytesIO(image_str)).convert("RGB")

        image = self.img_transform(image)

        return image, is_valid

    def prepare_empty_image(self, num_empty=0):
        if num_empty == 0:
            return []

        image_str = self.empty_image
        image = Image.open(io.BytesIO(image_str)).convert("RGB")

        return [self.img_transform(image) for _ in range(num_empty)]

    def prepare_multiple_images(self, images, max_num=3):
        images_out = []
        valid_out = []

        for img in images:
            if img == "":
                continue

            image, is_valid = self.prepare_single_image(img, is_url=True)

            if is_valid == 1:
                images_out.append(image)
                valid_out.append(is_valid)

            if len(images_out) >= max_num:
                break

        if len(images_out) < max_num:
            empty_num = max_num - len(images_out)
            images_out.extend(self.prepare_empty_image(empty_num))
            valid_out.extend([0] * empty_num)

        return images_out, valid_out

    def prepare_single_text(self, text, max_len):
        is_valid = 1
        if text == "":
            is_valid = 0

        if self.cutter_enable:
            tokens = []
            for word in self.cutter.cut(text):
                tokens.extend(self.tokenizer(word))
            tokens = tokens[: max_len - 2]
            token_ids = [self.vocab[t] for t in tokens]
            token_ids = [self.CLS] + token_ids + [self.SEP]
            token_ids = token_ids + [self.PAD] * (max_len - len(token_ids))
        else:
            tokens = self.tokenizer.tokenize(text)[: max_len - 2]
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_ids = token_ids + [self.PAD] * (max_len - len(token_ids))  # 填充至最大长度

        return token_ids, is_valid

    def prepare_multiple_texts(self, texts, max_len):
        texts_out = []
        valid_out = []

        for text in texts:
            token_ids, is_valid = self.prepare_single_text(text, max_len)
            texts_out.append(token_ids)
            valid_out.append(is_valid)

        return texts_out, valid_out

    def transform(self, data_dict: dict):
        main_images, main_image_masks = self.prepare_multiple_images(data_dict["main_images"], self.max_main)
        desc_images, desc_image_masks = self.prepare_multiple_images(data_dict["desc_images"], self.max_desc)
        sku_images, sku_image_masks = self.prepare_multiple_images(data_dict["sku_images"], self.max_sku)

        main_ocrs = data_dict["main_ocr"][: self.max_main]
        main_ocrs = main_ocrs + [""] * (self.max_main - len(main_ocrs))
        main_ocrs, main_text_masks = self.prepare_multiple_texts(main_ocrs, self.max_len)

        desc_ocrs = data_dict["desc_ocr"][: self.max_desc]
        desc_ocrs = desc_ocrs + [""] * (self.max_desc - len(desc_ocrs))
        desc_ocrs, desc_text_masks = self.prepare_multiple_texts(desc_ocrs, self.max_len)

        sku_ocrs = data_dict["sku_ocr"][: self.max_sku]
        sku_ocrs = sku_ocrs + [""] * (self.max_sku - len(sku_ocrs))
        sku_ocrs, sku_text_masks = self.prepare_multiple_texts(sku_ocrs, self.max_len)

        product_name, product_name_masks = self.prepare_single_text(data_dict["product_name"], self.max_len)

        if len(self.ner_tasks) > 0:
            other_text = "sku名称：" + data_dict["product_sku"] + ";" + "商店名称：" + data_dict["shop_name"]
        else:
            other_text = (
                "sku名称："
                + data_dict["product_sku"]
                + ";"
                + "商店名称："
                + data_dict["shop_name"]
                + ";"
                + "商品属性："
                + ",".join(["{}:{}".format(k, v) for k, v in data_dict["product_format_new"].items()])
                + ";"
                + "ner："
                + ",".join(["{}:{}".format(k, v) for k, v in data_dict["ner"].items()])
            )
        other_text, other_text_masks = self.prepare_single_text(other_text, self.max_len)

        # Category label
        level1_cid = str(data_dict["first_cid_new"])
        level2_cid = str(data_dict["second_cid_new"])
        label = 0
        label_l1 = 0
        if level1_cid in self.category_dict["level1"]["id2idx"]:
            label_l1 = self.category_dict["level1"]["id2idx"][level1_cid] + 1
        if level2_cid in self.category_dict["level2"]["id2idx"]:
            label = self.category_dict["level2"]["id2idx"][level2_cid] + 1

        result_dict = {
            "main_images": torch.stack(main_images, dim=0),
            "main_image_masks": main_image_masks,
            "desc_images": torch.stack(desc_images, dim=0),
            "desc_image_masks": desc_image_masks,
            "sku_images": torch.stack(sku_images, dim=0),
            "sku_image_masks": sku_image_masks,
            "main_ocrs": main_ocrs,
            "main_text_masks": main_text_masks,
            "desc_ocrs": desc_ocrs,
            "desc_text_masks": desc_text_masks,
            "sku_ocrs": sku_ocrs,
            "sku_text_masks": sku_text_masks,
            "product_name": product_name,
            "product_name_masks": product_name_masks,
            "other_text": other_text,
            "other_text_masks": other_text_masks,
            "label": label,
            "label_l1": label_l1,
        }

        # Ner or Format task label
        # there may be multiple ner labels
        # there must be single format labels
        for ner_index, task in enumerate(self.ner_tasks):
            ner_key = "ner_{}".format(ner_index)
            ner_label = np.zeros(
                shape=(len(self.ner_task_dict[task]["label2idx"]) + 1,),
                dtype=np.float32,
            )  # index 0 denotes other labels
            if task not in data_dict["ner"]:
                ner_label[0] = 1.0
            else:
                has_label = False
                for ner_val in data_dict["ner"][task].split("||"):
                    if ner_val in self.ner_task_dict[task]["label2idx"]:
                        ner_label[self.ner_task_dict[task]["label2idx"][ner_val] + 1] = 1.0
                        has_label = True
                if has_label:
                    ner_label /= ner_label.sum()  # Normalize to 0~1
                else:
                    ner_label[0] = 1.0
            result_dict[ner_key] = ner_label

        return result_dict

    def batch_transform(self, data):
        keys = list(data[0].keys())
        batch = {k: [] for k in keys}

        for i, ibatch in enumerate(data):
            for k in keys:
                batch[k].append(ibatch[k])

        for k in keys:
            if isinstance(batch[k][0], torch.Tensor):
                batch[k] = torch.stack(batch[k], dim=0)
            elif isinstance(batch[k][0], np.ndarray):
                batch[k] = torch.from_numpy(np.stack(batch[k], axis=0))
            else:
                batch[k] = torch.tensor(batch[k])

        return batch


"""
DataModule for FashionProduct
"""


class FPDataModule(CruiseDataModule):
    def __init__(
        self,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        paths: Union[
            str, List[str]
        ] = "hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/product_pretrain_train_url",
        data_size: int = 64000000,
        val_step: int = 20,
        num_workers: int = 24,
        max_len: int = 128,
        pretrained_model_dir: str = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/asr_model/zh_deberta_base_l6_emd_20210720",
        local_pretrained_model_dir_prefix="/opt/tiger/liuyuhang/ckpt/",
        cutter_enable: bool = False,
        cutter_resource_dir: str = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/asr_model/libcut_data_zh_20200827fix2/",
        local_cutter_dir_prefix: str = "/opt/tiger/liuyuhang/cutter/",
        category_path="hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/category_dict_pt.json",
        ner_task_dict="hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/ner_format_statistics/ner_task_dict.json",
        ner_tasks=["商品", "颜色", "风格", "材质", "样式"],
        max_main=3,
        max_desc=5,
        max_sku=3,
    ):
        super().__init__()
        self.save_hparams()

        """
        Prepare Deberta vocab
        """
        suffix = self.hparams.pretrained_model_dir.strip("/").split("/")[-1]
        self.local_pretrained_model_dir = f"{self.hparams.local_pretrained_model_dir_prefix}/{suffix}"

        """
        Prepare cutter
        """
        if self.hparams.cutter_enable:
            suffix = self.hparams.cutter_resource_dir.strip("/").split("/")[-1]
            self.local_cutter_dir = f"{self.hparams.local_cutter_dir_prefix}/{suffix}"
        else:
            self.local_cutter_dir = ""

    def local_rank_zero_prepare(self) -> None:
        # download cutter resource
        if self.hparams.cutter_enable:
            if not os.path.exists(self.local_cutter_dir):
                os.makedirs(self.hparams.local_cutter_dir_prefix, exist_ok=True)
                os.system(f"hdfs dfs -copyToLocal {self.hparams.cutter_resource_dir} {self.local_cutter_dir}")

    def setup(self, stage) -> None:
        paths = self.hparams.paths
        if isinstance(paths, str):
            paths = [paths]

        """
        Split data into train/val
        """
        files = hlist_files(paths)
        if not files:
            raise RuntimeError(f"No valid files can be found matching `paths`: {train_paths}")

        # use the last file as validation
        self.train_files = files
        self.val_files = files

    def train_dataloader(self):
        return DistributedCruiseDataLoader(
            data_sources=[self.train_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.train_batch_size],
            num_workers=self.hparams.num_workers,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=FPImageTextProcessor(
                mode="train",
                vocab_path=self.local_pretrained_model_dir + "/vocab.txt",
                cutter_enable=self.hparams.cutter_enable,
                cutter=self.local_cutter_dir,
                max_len=self.hparams.max_len,
                category_path=self.hparams.category_path,
                ner_task_dict=self.hparams.ner_task_dict,
                ner_tasks=self.hparams.ner_tasks,
                max_main=self.hparams.max_main,
                max_desc=self.hparams.max_desc,
                max_sku=self.hparams.max_sku,
            ),
            predefined_steps=self.hparams.data_size // self.hparams.train_batch_size // self.trainer.world_size,
            source_types=["jsonl"],
            shuffle=True,
        )

    def val_dataloader(self):
        return DistributedCruiseDataLoader(
            data_sources=[self.val_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.val_batch_size],
            num_workers=self.hparams.num_workers // 2,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=FPImageTextProcessor(
                mode="val",
                vocab_path=self.local_pretrained_model_dir + "/vocab.txt",
                cutter_enable=self.hparams.cutter_enable,
                cutter=self.local_cutter_dir,
                max_len=self.hparams.max_len,
                category_path=self.hparams.category_path,
                ner_task_dict=self.hparams.ner_task_dict,
                ner_tasks=self.hparams.ner_tasks,
                max_main=self.hparams.max_main,
                max_desc=self.hparams.max_desc,
                max_sku=self.hparams.max_sku,
            ),
            predefined_steps=self.hparams.val_step,
            source_types=["jsonl"],
            shuffle=False,
        )
