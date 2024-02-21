import logging
from typing import List, Tuple

import fermion_ops
import torch
import util as utils

OUTPUT_DIR = "output/text_model_example"  # 全图导出的pt文件保存在这个路径下


def load_sample_service_input() -> List[Tuple]:
    test1 = "请问这个正常发货吗,这个是预售还是不是"
    test2 = "开关不好用,开关无论按下去还是不按都通电的"
    return [(test1,), (test2,)]


class FermionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = fermion_ops.BertTokenizerOp(  # 相关参数和你的Tokenizer参数保持一致
            vocab_file="vocab.txt",
            max_seq_length=32,
            erase_newline_whitespace=False,
            to_lower=True,
            tokenize_chinese_chars=False,
            greedy_sharp=True,
        )
        # 提前trace BERT模型得到的model_cuda.jit，因此这里定义模型的时候直接使用torch.jit.load即可
        self.model = torch.jit.load("model_cuda.jit")
        self.model.cuda()
        self.model.eval()

    @torch.jit.export
    # 注意这里静态类型的数量要和返回的变量保持一致
    def preprocess_single(self, text: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids, input_mask, segment_ids = self.tokenizer(text)
        return (
            input_ids,
            input_mask,
            segment_ids,
        )

    @torch.jit.export
    def infer_batched(
        self, input_ids: torch.Tensor, input_mask: torch.Tensor, segment_ids: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        tensor = self.model(input_ids, input_mask, segment_ids)
        return (tensor.to(torch.float32),)

    def forward(self, text: List[str]) -> Tuple[torch.Tensor]:
        (
            input_ids,
            input_mask,
            segment_ids,
        ) = self.preprocess_single(text)
        (ret,) = self.infer_batched(input_ids.cuda(), input_mask.cuda(), segment_ids.cuda())
        return (ret,)  # 后面服务调用的时候需要这个变量名


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

    # output = full_model(
    #     [
    #         "请问这个正常发货吗,这个是预售还是不是",
    #         "开关不好用,开关无论按下去还是不按都通电的",
    #     ]
    # )
    # print(output)
