# https://github.com/htdt/hyp_metric/blob/master/proxy_anchor/dataset/SOP.py

import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image

from .base import *


class SOP(BaseDataset):
    def __init__(self, root, mode, train_split="Ebay_train.txt", transform=None):
        self.root = root + "/Stanford_Online_Products"
        self.mode = mode
        self.transform = transform
        if self.mode == "train":
            self.classes = range(0, 11318)
        elif self.mode == "eval":
            self.classes = range(11318, 22634)

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        metadata = open(
            os.path.join(
                self.root,
                train_split if self.classes == range(0, 11318) else "Ebay_test.txt",
            )
        )
        for i, (image_id, class_id, _, path) in enumerate(map(str.split, metadata)):
            if i > 0:
                if int(class_id) - 1 in self.classes:
                    self.ys += [int(class_id) - 1]
                    self.I += [int(image_id) - 1]
                    self.im_paths.append(os.path.join(self.root, path))


class MySOP(torch.utils.data.Dataset):
    def __init__(self, root, mode, train_split="Ebay_train.txt", transform=None):
        self.root = root + "/Stanford_Online_Products"
        self.mode = mode
        self.transform = transform
        self.classes = []
        self.im_paths = []

        assert os.path.exists(os.path.join(self.root, train_split))
        metadata = open(
            os.path.join(
                self.root,
                train_split if self.mode == "train" else "Ebay_test.txt",
            )
        )
        for i, (_, class_id, _, path) in enumerate(map(str.split, metadata)):
            if i > 0:
                self.im_paths.append(os.path.join(self.root, path))
                self.classes.append(int(class_id))

    def __getitem__(self, index):
        # Load the image
        try:
            im = Image.open(self.im_paths[index]).convert("RGB")
        except:
            index = random.randint(0, len(self.im_paths))
            im = Image.open(self.im_paths[index]).convert("RGB")
            # random_img = np.random.rand(384, 384, 3) * 255
            # im = Image.fromarray(np.uint8(random_img))
        if self.transform is not None:
            im = self.transform(im)

        label = self.classes[index]
        return im, label

    def __len__(self):
        return len(self.im_paths)
