from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union

from PIL import Image
from torchvision.transforms import transforms

from ...processor_utils import CENTER_IMAGE, RESIZE_IMAGE, ImageInput, ProcessorImageBase


@dataclass
class FalBertImageProcessor(ProcessorImageBase):
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    size: int = RESIZE_IMAGE
    center: int = CENTER_IMAGE
    frame_length: int = 8
    test_mode: bool = False

    def __post_init__(self):
        super().__init__(**self.__dict__)
        self.transform = self.get_transforms()

    def get_transforms(self) -> Callable:
        """get a image transforms combination

        Returns
        -------
        Callable
            a class that can be called
        """
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        if not self.test_mode:
            com_transforms = transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.center),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            com_transforms = transforms.Compose(
                [
                    transforms.Resize(self.size),
                    transforms.CenterCrop(self.center),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

        return com_transforms

    def preprocess(self, image: Union[str, ImageInput, List[str]], **kwds: Any):
        """preprocess image

        Parameters
        ----------
        image : Union[str, ImageInput]
            the url or array-like of the target image

        Returns
        -------
        _type_
            _description_
        """

        def _convert(image_url):
            try:
                image_ = Image.open(image_url).convert("RGB")
            except:  # noqa: E722
                image_ = Image.new("RGB", (self.size, self.size), (255, 255, 255))

            return self.transform(image_)

        if isinstance(image, str):
            return _convert(image)
        elif isinstance(image, list):
            return list(map(_convert, image[: self.frame_length]))
        else:
            return self.transform(image)
