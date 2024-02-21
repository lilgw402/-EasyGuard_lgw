import time
from typing import Dict, List, Union

import torch
import xperf_gpt
from transformers import LlamaForCausalLM, LlamaTokenizer

xperf_gpt.load_xperf_gpt()


class EndpointHandler:
    def __init__(self, ckpt_path="chinese-alpaca-2-13b-1008/"):
        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip
        self.model = LlamaForCausalLM.from_pretrained(
            ckpt_path,
            torch_dtype=torch.float16,
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(ckpt_path)
        self.model.eval()
        self.xperf_generator = xperf_gpt.init_inference(
            self.model,
            max_batch_size=2,  # 会根据最大配置预先分配显存，按需调整
            max_length=1024,
            use_xperf_gpt=True,
            do_script=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            top_k=1,
            max_new_tokens=80,
        )
        warm_text = {
            "text": [
                "请结合带货列表判断抖音电商的直播话术违反了哪些管控规则，并输出违规区域\n直播话术:'''第0句话：家电的王者，原来很多户外的装备，家用小的这种电器，您平时在其他家里面用的什么品牌？\n第1句话：这种电器很可能都是从我们厂子出去的，咱们都是做国际大牌代加工的。\n第2句话：这个品质您放心拿回去用，放心拿回去看对不对？\n第3句话：对的，免费拿回家用七天用的好了，您留下用的不好。\n第4句话：<br/>随时可以给主播退回来的，主播不要您一分钱一分钱都不要对不对？\n第5句话：对的，刚刚所有跟主播互动的哥哥姐姐叔叔阿姨爷爷奶奶们，如果说您感兴趣的话，如果说主播给您解答到位的话，您可以放心去拍，放心去入好不好？\n第6句话：好了，咱们还有最后的一单哈。\n第7句话：还有最后一单对不对？\n第8句话：好的，欢迎这个月辉啊！\n第9句话：感谢月辉的关注！\n第10句话：<br/>感谢美好时光的关注，您能给主播点关注的话，主播非常感谢您的支持，无以言报。\n第11句话：所有给主播点了关注的主播，祝您一生平安，祝您家人和和美美好不好？\n第12句话：好的，晚上好。\n第13句话：晚上好，咱们今天如果说我点上了吗？\n第14句话：呃，应该是点关注和粉丝灯呢？\n第15句话：粉丝灯牌对不对？\n第16句话：没有看到，没有看到，你可以点一下。\n第17句话：'''\n带货列表:'''0号商品，一级类目名称为汽车用品，1号商品，一级类目名称为生活电器，2号商品，一级类目名称为3C数码配件，3号商品，一级类目名称为电子/电工'''\n请输出违反的管控规则，以及违规区域:\n".encode()  # noqa: E501
            ]
        }
        self(warm_text)
        self(warm_text)
        self(warm_text)

    def process_batch(self, batch_list, max_length):
        input_ids_list = []
        attention_mask_list = []
        for sample in batch_list:
            input_ids = sample["input_ids"]
            attention_mask = sample["attention_mask"]
            text_length = input_ids.shape[1]
            if text_length < max_length:
                pad_tensor = torch.LongTensor([self.tokenizer.pad_token_id] * (max_length - text_length)).reshape(
                    1, -1
                )
                mask_tensor = torch.LongTensor([0] * (max_length - text_length)).reshape(1, -1)
                input_ids = torch.cat([pad_tensor, input_ids], 1)
                attention_mask = torch.cat([mask_tensor, attention_mask], 1)

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
        input_ids = torch.cat(input_ids_list, 0)
        attention_mask = torch.cat(attention_mask_list, 0)
        return input_ids, attention_mask

    def __call__(
        self,
        request: Dict[str, Union[List[bytes], List[int], List[float]]],
    ):
        start = time.time()
        texts = [text.decode() for text in request["text"]]
        batch_list = []
        max_length = 0
        for _, text in enumerate(texts):
            inputs = self.tokenizer(text, return_tensors="pt")
            if inputs["input_ids"].shape[1] > max_length:
                max_length = inputs["input_ids"].shape[1]
            batch_list.append(inputs)

        batch_input_ids, batch_mask = self.process_batch(
            batch_list,
            max_length,
        )
        print("data proprecess: ", time.time() - start)
        start = time.time()
        batch_input_ids = batch_input_ids.cuda()
        batch_mask = batch_mask.cuda()
        length = batch_input_ids.size(1)
        output = self.xperf_generator(
            batch_input_ids,
            attention_mask=batch_mask,
        )
        output_text_list = []
        for index in range(len(batch_list)):
            output_text = self.tokenizer.decode(output[index][length:])
            output_text_list.append(output_text.encode())
        spend_time = time.time() - start
        print("each infer time: ", spend_time)
        return dict(generate=output_text_list)


if __name__ == "__main__":
    endpoint = EndpointHandler()
    warm_text = {
        "text": [
            "请结合带货列表判断抖音电商的直播话术违反了哪些管控规则，并输出违规区域\n直播话术:'''第0句话：家电的王者，原来很多户外的装备，家用小的这种电器，您平时在其他家里面用的什么品牌？\n第1句话：这种电器很可能都是从我们厂子出去的，咱们都是做国际大牌代加工的。\n第2句话：这个品质您放心拿回去用，放心拿回去看对不对？\n第3句话：对的，免费拿回家用七天用的好了，您留下用的不好。\n第4句话：<br/>随时可以给主播退回来的，主播不要您一分钱一分钱都不要对不对？\n第5句话：对的，刚刚所有跟主播互动的哥哥姐姐叔叔阿姨爷爷奶奶们，如果说您感兴趣的话，如果说主播给您解答到位的话，您可以放心去拍，放心去入好不好？\n第6句话：好了，咱们还有最后的一单哈。\n第7句话：还有最后一单对不对？\n第8句话：好的，欢迎这个月辉啊！\n第9句话：感谢月辉的关注！\n第10句话：<br/>感谢美好时光的关注，您能给主播点关注的话，主播非常感谢您的支持，无以言报。\n第11句话：所有给主播点了关注的主播，祝您一生平安，祝您家人和和美美好不好？\n第12句话：好的，晚上好。\n第13句话：晚上好，咱们今天如果说我点上了吗？\n第14句话：呃，应该是点关注和粉丝灯呢？\n第15句话：粉丝灯牌对不对？\n第16句话：没有看到，没有看到，你可以点一下。\n第17句话：'''\n带货列表:'''0号商品，一级类目名称为汽车用品，1号商品，一级类目名称为生活电器，2号商品，一级类目名称为3C数码配件，3号商品，一级类目名称为电子/电工'''\n请输出违反的管控规则，以及违规区域:\n".encode()  # noqa: E501
        ]
        * 1
    }
    endpoint(request=warm_text)
    endpoint(request=warm_text)
    start = time.time()
    for i in range(40):
        results = endpoint(request=warm_text)
        results = results["generate"][0].decode()
        print(results)
    print("spend time: ", (time.time() - start) / 40)
