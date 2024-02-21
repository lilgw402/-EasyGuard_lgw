# https://github.com/htdt/hyp_metric/blob/master/proxy_anchor/dataset/SOP.py

import os
import random

import torch
from PIL import Image


class Products10k(torch.utils.data.Dataset):
    def __init__(self, root, mode, train_split="train_list.txt", transform=None):
        self.root = root + "/Products-10k"
        self.mode = mode
        self.transform = transform
        self.classes = []
        self.im_paths = []

        assert os.path.exists(os.path.join(self.root, train_split))
        metadata = open(
            os.path.join(
                self.root,
                train_split if self.mode == "train" else "val_list_dummy.txt",
            )
        )
        for i, (path, class_id, _) in enumerate(map(str.split, metadata)):
            self.im_paths.append(os.path.join(self.root, path))
            self.classes.append(int(class_id))

    def __getitem__(self, index):
        # Load the image
        try:
            im = Image.open(self.im_paths[index]).convert("RGB")
        except:  # noqa: E722
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
