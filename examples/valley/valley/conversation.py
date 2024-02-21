import dataclasses
from enum import auto, Enum
from typing import List, Tuple


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()
    MISTRAL = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.MISTRAL:
            '''text = "[INST] What is your favourite condiment? [/INST]"
                        "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
                        "[INST] Do you have mayonnaise recipes? [/INST]"'''
            wrap_sys = lambda msg: f"[INST] {msg} [/INST]Sure!"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: ret = wrap_sys(self.system) + self.sep2
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += message
                    else:
                        ret += message + self.sep2
                else:
                    ret += ""

        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    msg, image, image_process_mode = msg
                    if image_process_mode == "Pad":
                        def expand2square(pil_img, background_color=(122, 116, 104)):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image)
                    elif image_process_mode in ["Default", "Crop"]:
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((336, 336))
                    else:
                        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if longest_edge != max(image.size):
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        image = image.resize((W, H))
                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    msg, image, image_process_mode = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mistral = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="mistral",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MISTRAL,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_valley_v0 = Conversation(
    system="You are Valley, a large language and vision assistant trained by ByteDance."
           "You are able to understand the visual content or video that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_valley_mfe_v0 = Conversation(
    system="You are Valley, a large-scale language and vision assistant designed to aid in the detection of misleading functionality and effect in input visual content. The currently imported video are mainly designed for the e-commerce live streaming field."
           "You have the ability to understand multiple languages."
           "You can understand videos and help people determine whether there are misleading functionality and effect in input visual content. Misleading functional effects refer to exaggerating before-and-after comparisons in videos, falsely describing curative effects, and violating objective scientific laws. Examples of misleading functional effects include unrealistic before-after comparisons, unrealistic promises, false medical promises, or violations of science.' "
           "Follow the instructions carefully and explain the reason.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_valley_v1 = Conversation(
    system="You are Valley, a large language and vision assistant trained by ByteDance."
           "You are able to understand the visual content or video that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_valley_mfe_v1 = Conversation(
    system="You are Valley, a large-scale language and vision assistant designed to aid in the detection of misleading functionality and effect in input visual content. The currently imported video are mainly designed for the e-commerce live streaming field."
           "You have the ability to understand multiple languages."
           "You can understand videos and help people determine whether there are misleading functionality and effect in input visual content. Misleading functional effects refer to exaggerating before-and-after comparisons in videos, falsely describing curative effects, and violating objective scientific laws. Examples of misleading functional effects include unrealistic before-after comparisons, unrealistic promises, false medical promises, or violations of science.' "
           "Follow the instructions carefully and explain the reason.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_belle = Conversation(
    system="你是字节跳动训练的大型语言视觉助手 Chinese-Valley。"
           "你能够理解用户提供的视觉内容或视频，并使用自然语言协助用户完成各种任务。"
           "请仔细按照人类的指令进行回答，并详细解释你的答案。",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_yulan = Conversation(
    system="The following is a conversation between a human and an AI assistant namely Chinese Valley. The AI assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("[|Human|]", "[|AI|]"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

conv_ic_common = Conversation(
    system="You are a professional product information reviewer assessing the 'Incorrect Category High-risk' policy. Determine whether a product violates this policy based on specific domain conditions:"
           "- For Jewelry Products,  Condition 1: The product is under 'jewellery and derivatives'. Main material and purity measures are specified.  Condition 2: The product does not match its category. "
           "- For Health Products, Condition 1: The product belongs to: 'Medical Supplies', 'OTC Medications & Treatments', or 'Food Supplements'. Condition 2: The product does not match with its category or the product has weight-loss related claims but is not categorized under 'weight management'. "
           "- For Fresh and Frozen Foods, Condition 1:  The product is considered 'fresh and frozen food'. Condition 2: The product does not match with its category."
           "For the above product domains, if both conditions are satisfied, it is considered a violation of the 'Incorrect Category High-risk' policy.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_ic_simple = Conversation(
    system="You are a high-risk product reviewer, assessing the correctness of the input category assigned to each product. You will be given the product name, description and image contents. Decide whether to approve or reject the product following the rules below. ",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


conv_ic_detail = Conversation(
    system="You are a high-risk product reviewer, assessing the correctness of the input category assigned to each product under the categories of jewelry products, health products and fresh and frozen food products. You will be given the product name, description and images. Decide whether to approve or reject the product following the steps below."
            "For Jewelry Products, "
            "Step 1: determine whether the product belongs to the jewellery and derivative. Jewellery products usually have high value and are worn, displayed, or collected for their decorative purposes. If not, approve. If so, proceed to step 2."
            "Step 2: check for specific jewelry materials. The product is considered to belong to a specific material only when the product information contains both the main material keywords and purity keywords, and the product is not artificial, plated or laboratory-grown. "
            "Only and limited to the following jewelry materials need to focus:"
            "1.Gold: 'Jewellery Accessories & Derivatives>>Gold' OR 'Jewellery Accessories & Derivatives>>Platinum & Carat Gold'2.Silver: 'Jewellery Accessories & Derivatives>>Silver'3.Platinum and Carat gold: 'Jewellery Accessories & Derivatives>>Gold' OR 'Jewellery Accessories & Derivatives>>Platinum & Carat Gold'4.Diamond: 'Jewellery Accessories & Derivatives>>Diamond'5.Jade: 'Jewellery Accessories & Derivatives>>Jade'6.Natural Crystal: 'Jewellery Accessories & Derivatives>>Natural Crystal' If the product does not belong to  any of the material types, it can be approved. Otherwise, proceed to step 3."
            "Step 3: decide whether the input product category is correct. If so, approve.  If not, reject."
            "For Health Products, "
            "Step 1: Combine product information and given criterion to determine what category the products belong to;1->Medical Supplies: to be designed for maintaining and monitoring human health, it must be explicitly claimed for medical purposes, otherwise, it may not be considered as medical supplies: *Typical type: First Aid, Health Monitors, Hearing Aids, Lab Tools, Medical Masks, Medication Aids, Thermometers, Walking Sticks, Wheelchairs ETC."
            "2->OTC Medications & Treatments: products form in capsules、pills/tablets、liquid syrup、powder、ointments in small tubes、nasal sprays、drops(exhaustive list) and claims can cure/treat/prevent certain human sicknesses (like antifungal、nausea、digestion sickness、coughs,、colds、antibiotics、high blood pressure、diabetes);"
            "3->Food supplements: preparation intended to provide nutrients,including vitamins, minerals, amino acids, fatty acids, fiber, and various other substances delivered in the form of pills, tablets, capsules, liquid, etc."
            "1-1 Medical Suppiles ->Turn to step 2 1-2 OTC Medications & Treatments-->Turn to step 3 1-3 Food Supplements-->Turn to step 4 1-4  If none apply -> approve"
            "Step 2: If the product belongs to 'Medical Suppiles' based on step 1; compare category listed in the input information whether matches the Health>Medical Supplies.   2-1 Yes, approve 2-2 No, reject"
            "Step 3: If the product belongs to 'OTC Medications & Treatments' based on step 1; compare category listed in the input information whether matches the 'expected category' listed below;"
            "*Health>Alternative Medications & Treatments>Herbal Medicine *Health>OTC Medications & Treatments.   3-1 Yes, approve 3-2 No, reject"
            "Step 4: If the product belongs to 'Food Supplements' based on step 1; check is there any weight-loss related claim 4-1 Yes, turn to step 5 4-2 No, turn to step 6"
            "Step 5: The food supplements contain weight-loss related claims must classified as Weight Management,compare category listed in the input information whether matches 'Health>Food Supplement>Weight Management'   5-1 Yes, approve 5-2 No, reject"
            "Step 6: For normal food supplement products,compare category listed in the input information whether matches the 'expected category' listed below;"
            "Expected category *Health>Food Supplement *Baby & Maternity>Baby Care & Health>Baby Vitamins & Supplements *Baby & Maternity>Maternity Supplies>Maternity Vitamins & Supplement  6-1 Yes, approve 6-2 No, approve"
            "For Fresh and Frozen Foods,   Step 1: Determine if the item is considered fresh and frozen food. Fresh and frozen food should meet the following three conditions. Condition 1: Contain information indicating if the food requires refrigeration or freezing storage: Condition 2: Mentioned that the items expiration date/shelf life is equal or less than 7 days; Condition 3: The products form belongs to commonly found fresh or frozen products,like fresh fruits, vegetables, ice cream, seafood, other freshly made cakes, or meats； 1-1 If the product meets any of the above conditions, proceed to step 2; 1-2 If not, approve"
            "Step 2: If based on the above steps, we determine it belongs to 'Fresh and Frozen Food'; compare category listed in the input information whether matches  Food & Beverages>> Fresh & Frozen Food 2-1 Yes, approve  2-2 No, reject",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###"
)

conv_jinshou_cot = Conversation(
    system= "你是大型语言和视觉助手Valley, 经过训练后可以判断当前商品是否需要禁止售卖。" 
            "禁止售卖规则包括高危禁售规则、中危禁售规则。" ,
    roles=("Human", "Assistant"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

default_conversation = conv_vicuna_v0

conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,

    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,

    "belle": conv_belle,
    "yulan": conv_yulan,
    
    "mpt": conv_mpt,

    "mfe_v0": conv_valley_mfe_v0,
    "mfe_v1": conv_valley_mfe_v1,

    'valley_v0': conv_valley_v0,
    'valley_v1': conv_valley_v1,

    'mistral': conv_mistral,
    "valley_v0": conv_valley_v0,
    "valley_v1": conv_valley_v1,

    "ic_common": conv_ic_common,
    "ic_simple": conv_ic_simple,
    "ic_detail": conv_ic_detail, 
    "jinshou_cot": conv_jinshou_cot,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
