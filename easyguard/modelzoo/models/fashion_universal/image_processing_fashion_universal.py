from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch
from torchvision.transforms import transforms

from ...processor_utils import CENTER_IMAGE, RESIZE_IMAGE, ImageInput, ProcessorImageBase


@dataclass
class FashionUniversalProcessor(ProcessorImageBase):
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    size: Optional[float] = RESIZE_IMAGE
    center: Optional[float] = CENTER_IMAGE

    def __post_init__(self):
        super().__init__(**self.__dict__)
        self.transform = self.get_transforms()

    def get_transforms(self) -> Callable:
        """if the transformation of image is fixed, it is better to use `transforms.Compose` to get a callable class"""

    def preprocess(self, image: ImageInput, **kwds: Any):
        """preprocess image

        Parameters
        ----------
        image : ImageInput
            the target image

        Returns
        -------
        _type_
            _description_
        """
        if not self.transform:
            resize = transforms.Resize(self.size)
            center_crop = transforms.CenterCrop(self.center)
            to_tensor = transforms.ToTensor() if not torch.is_tensor(image) else None
            normalize = transforms.Normalize(mean=self.mean, std=self.std)

            funcs = [
                resize,
                center_crop,
                to_tensor,
                normalize,
            ]

            self.transform = transforms.Compose([_ for _ in funcs if _])

        return self.transform(image)
