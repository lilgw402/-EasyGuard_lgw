# https://github.com/htdt/hyp_metric/blob/master/proxy_anchor/dataset/SOP.py

from .base import *


class BOP(BaseDataset):
    def __init__(self, root, mode, transform=None):
        self.root = root + "/Byte_Online_Products"
        self.mode = mode
        self.transform = transform
        self.classes = []

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        metadata = open(os.path.join(self.root, "Byte_test.txt"))
        for i, (image_id, class_id, _, path) in enumerate(map(str.split, metadata)):
            if i > 0:
                self.ys += [int(class_id) - 1]
                self.I += [int(image_id) - 1]
                self.im_paths.append(os.path.join(self.root, path))
                self.classes.append(int(class_id) - 1)
