import logging
from typing import List, Tuple

import fermion_ops
import torch
import torchvision
import util as utils

OUTPUT_DIR = "output/image_model_example"  # 全图导出的pt文件保存的路径


def load_sample_service_input() -> List[Tuple]:
    with open("./demo.jpg", "rb") as f:
        img = f.read()
    return [(img,)]


class FermionModel(torch.nn.Module):
    def __init__(
        self,
        resize_size=256,
        crop_size=224,
        norm_mean=(0.485, 0.456, 0.406),
        norm_std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        # ImageDecodeOp包含了解码和Resize两个操作，返回的是个Tensor
        self.decode = fermion_ops.ImageDecodeOp(
            bgra_alpha_blending=False,
            use_libjpeg_turbo=True,
            use_gpu=False,
            resize_width=resize_size,
            resize_height=resize_size,
            resize_method=fermion_ops.ResizeMethod.INTER_LINEAR,  # 注意这里的采样方式要和之前的采样方式对齐，否则数值会对不齐
            resize_convert_to_float=False,
        )
        # 直接使用torchvision的OP，支持输入是个Tensor
        self.crop = torchvision.transforms.CenterCrop(size=crop_size)
        self.norm = torchvision.transforms.Normalize(mean=norm_mean, std=norm_std)
        self.rescale_factor = 1 / 255.0
        self.transpose = [0, 3, 1, 2]
        # 不需要定义模型的结构，直接Load模型的pt文件
        self.model = torch.jit.load("trace_model_zhi_1701383336_half.pt")
        self.model.cuda()
        self.model.eval()

    @torch.jit.export
    def preprocess_single(self, img_list: List[str]) -> Tuple[torch.Tensor]:
        # img_list是个List，每个元素类型是byte
        x = self.decode(img_list, is_video=False)[0]
        return (x,)

    @torch.jit.export
    def infer_batched(self, tensor: torch.Tensor) -> Tuple[torch.Tensor]:
        # 这几处的前处理操作放在infer_batched函数下面而不是preprocess_single函数下面
        # 是因为它们是支持cuda操作的，可以加速
        tensor = tensor.permute(self.transpose)
        tensor = self.crop(tensor)
        tensor = tensor * self.rescale_factor
        tensor = self.norm(tensor)
        tensor = self.model(tensor.half())
        return (tensor,)

    def forward(self, image: List[str]) -> Tuple[torch.Tensor]:
        (_1,) = self.preprocess_single(image)
        (_3,) = self.infer_batched(_1.cuda())
        return (_3,)  # 后面服务调用的时候需要这个变量名


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    service_input: List = load_sample_service_input()
    full_model = FermionModel()

    output = utils.forward(full_model, service_input)
    if utils.save(full_model, service_input, OUTPUT_DIR):
        logging.info("Done saving. Successful Run!")
    else:
        logging.error("Fail when saving")

    import shutil

    shutil.copy(__file__, OUTPUT_DIR)
