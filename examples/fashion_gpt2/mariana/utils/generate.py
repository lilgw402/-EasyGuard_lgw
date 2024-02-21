"""

将一些解码的过程抽象出来
Copied from FUXI, 参考标准类似：https://huggingface.co/spaces/THUDM/GLM-130B

"""
import json
import sys
import traceback
import warnings
from builtins import print

import torch
from cruise.utilities.hdfs_io import hopen
from torch.nn import functional as F
from tqdm import tqdm

try:
    from promptsource.templates import DatasetTemplates, Template
except ImportError:
    warnings.warn("failed to load prompt source", ImportWarning)
try:
    from mariana.utils.processor import lambada_processor, ocnli_processor, rte_processor
except:
    from examples.fashion_gpt2.mariana.utils.processor import lambada_processor, ocnli_processor, rte_processor


@torch.no_grad()
def sample(
    model_decode_func,
    x,
    steps,
    temperature=1.0,
    do_sample=False,
    top_k=None,
    top_p=None,
    dynamic_top_p=None,
    omega=0.3,
    decay_lambda=0.9,
    eos=3,
    until_n_eos=1,
    full_stop_input_ids=None,
):
    """
    Take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time.

    函数实现了几个 sample strategy：
    1. Tempreature Sampling (通过 temperature 来控制)
    2. Top-K Sampling （通过 top_k 来控制）
    3. Top-P Sampling （通过 top_p 来控制）
    注意这里的实现是Top-K 和 Top-P 是互斥的

    model_decode_func: 解码函数，输入x，输出模型的解码logits（注意没有做softmax）
    x：输入，可以理解为前n个词。这里需要遍历 steps 次，每次把新 decode 出来的term 拼到 x 后面
    steps：解码的长度
    tempreature：对logits 做 / tempreature 的操作，tempreature 越小，logits 分布越陡峭，越确定性的预测高概率的term
    do_sample：是否有采样的过程，是的话，按probability 进行 multinomial 的采样，否则直接取最高概率的
    top_k：是否有 top k 的过程，top_k 表示 k 的个数。
    top_p：是否有 top p 的过程，top_p 表示 要高于的概率。
    eos：end of sentence token。当遇到这个 token 时，停止生成。
    until_n_eos：默认为1。有一些情况，我们希望多生成一些，这个参数用来控制直到的 n 次遇到 eos，才停止生成。

    默认支持的是单条。没测过batch level。

    """
    if top_k is not None and top_p is not None:
        raise ValueError("Either Top-K or Top-P, cannot choose both")
    if top_p is not None and dynamic_top_p is not None:
        raise ValueError("Either Top-P or Dynamic_Top-P, cannot choose both")
    if top_k is not None and dynamic_top_p is not None:
        raise ValueError("Either Top-K or Dynamic_Top-P, cannot choose both")

    meet_eos_count = 0
    k_w_reset = 0
    for k in range(steps):
        k_w_reset += 1
        logits = model_decode_func(x)
        # print("logits: {} with size: {}".format(logits, logits.size()))
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # print("logits after temp: {}".format(logits))
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # optionally crop probabilities to only the options whose prob larger than p
        elif top_p is not None:
            logits = top_p_logits(logits, top_p)
        elif dynamic_top_p is not None:
            decayed_top_p = max(omega, dynamic_top_p * (decay_lambda ** (k_w_reset - 1)))
            # print("omega:{}, decay_lambda: {}, step: {}, decayed_top_p: {}".format(omega, decay_lambda, k_w_reset-1, decayed_top_p))
            logits = top_p_logits(logits, decayed_top_p)
        # print("topk/topp logits: {} with size: {}".format(logits, logits.size()))
        probs = F.softmax(logits, dim=-1)
        # print(probs, probs.shape, 'probs')
        # sample from the distribution or take the most likely
        if do_sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        # print("ix: {}, full_stop_input_ids: {}".format(ix, full_stop_input_ids))
        if ix in full_stop_input_ids:
            k_w_reset = 0
            # print("ix: {}, full_stop_input_ids: {}, with reset_k: {}".format(ix, full_stop_input_ids, k_w_reset))

        if ix.tolist()[0][0] == eos:  # 遇到eos 就停止生成
            meet_eos_count += 1
            if meet_eos_count >= until_n_eos:
                break
        if x is not None:
            x = torch.cat((x, ix), dim=1)
        else:
            x = ix
        # print("x: {}".format(x))
    return x


@torch.no_grad()
def beam_search(
    model_decode_func,
    x,
    steps,
    beam_size=2,
    length_penalty_alpha=0.7,
    eos=3,
    until_n_eos=1,
):
    """
    Take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time.

    函数实现了beam search 的生成策略：
    model_decode_func: 解码函数，输入x，输出模型的解码logits（注意没有做softmax）
    x：输入，可以理解为前n个词。这里需要遍历 steps 次，每次把新 decode 出来的term 拼到 x 后面
    steps：解码的长度
    beam_size：beam search 长度
    eos：end of sentence token。当遇到这个 token 时，停止生成。
    until_n_eos：默认为1。有一些情况，我们希望多生成一些，这个参数用来控制直到的 n 次遇到 eos，才停止生成。

    默认支持的是单条。没测过batch level。

    """
    meet_eos_count = 0

    # 1 先做一次decode，取到 beam size 个引子
    first_logits = model_decode_func(x)
    best_k_probs, best_k_idx = first_logits[:, -1, :].topk(beam_size)
    beam_scores = torch.log(best_k_probs).view(beam_size)
    x = x.repeat([beam_size, 1])  # [beam_size, 1]
    x = torch.cat((x, best_k_idx[0].unsqueeze(-1)), dim=1)  # [beam_size, 2]

    # 2 开始遍历
    for k in range(steps):
        logits = model_decode_func(x)
        logits = logits[:, -1, :]

        # adopt the topk next-tokens for each sequence in the beam
        best_k2_probs, best_k2_idxs = torch.topk(logits, k=beam_size)  # (bsize_ * beam_size, beam_size)

        # compute the scores for k^2 sequences and select the best-k for the next beam
        beam_scores = beam_scores.unsqueeze(-1) + torch.log(
            best_k2_probs.view(beam_size, -1)
        )  # (beam_size, beam_size)

        beam_scores, best_k_idx_in_k2 = torch.topk(beam_scores.view(-1), k=beam_size)  # (bsize_, beam_size)

        # retrieve the prefex and next-token, and compose the sequence
        best_k_r_idxs, best_k_c_idxs = (
            best_k_idx_in_k2 // beam_size,
            best_k_idx_in_k2 % beam_size,
        )
        best_k_idx = best_k2_idxs[best_k_r_idxs, best_k_c_idxs]

        x = x[best_k_r_idxs]
        x = torch.cat((x, best_k_idx.unsqueeze(-1)), dim=1)  # [beam_size, seq_len + 1]

        # -- check if all beams contain eos
        eos_locs = x == eos  # (beam_size, seq_len)
        eos_num = eos_locs.sum(1)
        if eos_num.min() >= until_n_eos:
            break

        # # penalize scores by the sequence length and update the result sequence.
        # _, best_idxs = beam_scores.div(seq_lens.float() ** length_penalty_alpha).max(-1)  # (bsize_, )
        # # (bsize_, beam_size, slen_) -> (bsize_, slen_)
        # x = torch.gather(x, dim=1, index=best_idxs[...,None,None].expand(-1, 1, beam_seqs.size(-1))).squeeze()

    return x


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


def top_p_logits(logits, p):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # print("sorted_probs: {} with sorted_indices: {}".format(sorted_probs, sorted_indices))
    # cumulative_probs = torch.cumsum(F.softmax(sorted_probs, dim=-1), dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # print("cumulative_probs: {} with threshold: {}".format(cumulative_probs, p))
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    # print("sorted_indices_to_remove: {} ".format(sorted_indices_to_remove))
    indices_to_remove = torch.zeros_like(logits, dtype=sorted_indices_to_remove.dtype).scatter_(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )

    # indices_to_remove = sorted_indices[sorted_indices_to_remove]
    # print(indices_to_remove, indices_to_remove.shape, 'indices_to_remove')

    out = logits.clone()
    # print(out.shape, 'out')
    out[indices_to_remove] = -float("Inf")
    # print(out.shape, 'out after')

    return out


def get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
    assert len(scores.size()) == 1

    beam_size = self.beam_size

    # Get k candidates for each beam, k^2 candidates in total.
    best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

    # Include the previous scores.
    scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

    # Get the best k candidates from k^2 candidates.
    scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

    # Get the corresponding positions of the best k candidiates.
    best_k_r_idxs, best_k_c_idxs = (
        best_k_idx_in_k2 // beam_size,
        best_k_idx_in_k2 % beam_size,
    )
    best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

    # Copy the corresponding previous tokens.
    gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
    # Set the best tokens in this beam search step
    gen_seq[:, step] = best_k_idx

    return gen_seq, scores


@torch.no_grad()
def sample_generate(
    model,
    input_ids,
    steps=32,
    temperature=1.0,
    do_sample=False,
    top_k=None,
    top_p=None,
    dynamic_top_p=None,
    omega=0.3,
    decay_lambda=0.9,
    eos=5,
    until_n_eos=1,
    full_stop_input_ids=None,
):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """

    # 2. 构造 模型解码的func wrapper
    def model_decode_func(x):
        input_mask = torch.ones_like(x, device=x.device)
        res = model.decode(input_ids=x, input_mask=input_mask)
        return res["logits"]

    # 3. 生成结果
    decoded_sentence = sample(
        model_decode_func=model_decode_func,
        x=input_ids,
        steps=steps,
        temperature=temperature,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        dynamic_top_p=dynamic_top_p,
        omega=omega,
        decay_lambda=decay_lambda,
        eos=eos,
        until_n_eos=until_n_eos,
        full_stop_input_ids=full_stop_input_ids,
    )
    return decoded_sentence.tolist()[0]


def get_input_ids_of_stop_tokens(tokenizer, full_stop_tokens=".。！？！？"):
    full_stop_input_ids = tokenizer(full_stop_tokens)["input_ids"]
    print(
        "full_stop_tokens: {}, and corresponding full_stop_input_ids: {}".format(full_stop_tokens, full_stop_input_ids)
    )
    completion = "".join(tokenizer.decode(full_stop_input_ids))
    # print("full_stop_tokens after decode: {}".format(completion))
    return full_stop_input_ids


def play_console(
    tokenizer,
    model,
    trial_num=5,
    steps=256,
    temperature=0.6,
    do_sample=True,
    top_k=5,
    top_p=None,
    dynamic_top_p=None,
    omega=0.3,
    decay_lambda=0.9,
    until_n_eos=1,
):
    print("Input your prompt here...")
    full_stop_input_ids = get_input_ids_of_stop_tokens(tokenizer)
    for line in sys.stdin:
        try:
            text = line.strip()
            print("prompt: ", text)
            print("tokens: ", tokenizer.tokenize(text))
            input_ids = torch.tensor(tokenizer(text)["input_ids"]).unsqueeze(0).long()
            for i in range(trial_num):
                y = sample_generate(
                    model,
                    input_ids=input_ids.to(model.device),
                    steps=steps,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    dynamic_top_p=dynamic_top_p,
                    omega=omega,
                    decay_lambda=decay_lambda,
                    eos=tokenizer.eos_token_id,
                    until_n_eos=until_n_eos,
                    full_stop_input_ids=full_stop_input_ids,
                )
                # y = y.tolist()[0][1:]
                completion = "".join(tokenizer.decode(y))
                completion.replace("##", "")
                print(f"[{i}]: {completion}")
            print("Input your prompt here...")
        except Exception as e:
            print(e)
            print(traceback.format_exc())


def play_file(
    fname,
    tokenizer,
    model,
    output_file_path,
    trial_num=5,
    steps=256,
    temperature=0.6,
    do_sample=True,
    top_k=5,
    top_p=None,
    dynamic_top_p=None,
    omega=0.3,
    decay_lambda=0.9,
    until_n_eos=1,
    limit_samples=-1,
):
    count = 0
    with hopen(fname) as f, open(output_file_path, mode="w", encoding="utf-8") as fw:
        for line in f:
            count += 1
            if limit_samples > 0 and count >= limit_samples:
                print(f"Reach limit_samples: {limit_samples}, stop.")
                break
            try:
                jl = json.loads(line)
                text = jl["page_info"]["core_content"].strip()
                label = jl["label"].strip()
                input_ids = torch.tensor(tokenizer(text)["input_ids"]).unsqueeze(0).long()
                # print("input_ids: {}".format(input_ids))
                for i in range(trial_num):
                    y = sample_generate(
                        model,
                        input_ids=input_ids.to(model.device),
                        steps=steps,
                        temperature=temperature,
                        do_sample=do_sample,
                        top_k=top_k,
                        top_p=top_p,
                        dynamic_top_p=dynamic_top_p,
                        omega=omega,
                        decay_lambda=decay_lambda,
                        eos=tokenizer.eos_token_id,
                        until_n_eos=until_n_eos,
                    )
                    completion = "".join(tokenizer.decode(y))
                    completion.replace("##", "")
                    fw.write("\t".join([text, label, completion]) + "\n")
                    print(f"[{i}]: {completion}")
            except Exception as e:
                print(e)
                print(traceback.format_exc())


def play_file_qa(
    fname,
    tokenizer,
    model,
    output_file_path,
    trial_num=5,
    steps=256,
    temperature=0.6,
    do_sample=True,
    top_k=5,
    top_p=None,
    dynamic_top_p=None,
    omega=0.3,
    decay_lambda=0.9,
    until_n_eos=1,
    limit_samples=-1,
):
    print(f"Generating by prompts from {fname}...")
    full_stop_input_ids = get_input_ids_of_stop_tokens(tokenizer)
    with open(fname, mode="r", encoding="utf-8") as fr, open(output_file_path, mode="w", encoding="utf-8") as fw:
        for i, line in enumerate(tqdm(fr, desc=f"PROCESSING FILE: {fname}")):
            try:
                jl = json.loads(line)
                text = jl["page_info"]["query"].strip()
                label = jl["page_info"]["answer"]
                input_ids = torch.tensor(tokenizer(text)["input_ids"]).unsqueeze(0).long()
                for _ in range(trial_num):
                    y = sample_generate(
                        model,
                        input_ids=input_ids.to(model.device),
                        steps=steps,
                        temperature=temperature,
                        do_sample=do_sample,
                        top_k=top_k,
                        top_p=top_p,
                        dynamic_top_p=dynamic_top_p,
                        omega=omega,
                        decay_lambda=decay_lambda,
                        eos=tokenizer.eos_token_id,
                        until_n_eos=until_n_eos,
                        full_stop_input_ids=full_stop_input_ids,
                    )
                    completion = "".join(tokenizer.decode(y))
                    completion.replace("##", "")
                    fw.write("\t".join([text, label, completion]) + "\n")
            except Exception as e:
                print(e)
                print(traceback.format_exc())


def get_prompt_template(dataset_name, subset_name, template_name):
    dataset_name = dataset_name if dataset_name != "" else None
    subset_name = subset_name if subset_name != "" else None

    templates = DatasetTemplates(dataset_name, subset_name)
    prompt_template = templates[template_name]

    return prompt_template


def get_few_context(
    file_name,
    prompt_template,
    num_fewshot,
    subset_name,
    example_sep="\n###\n",
    prompt_target_sep=" ",
):
    examples = []
    with open(file_name, "r") as f:
        for line in f:
            data_dict = json.loads(line)
            label_id = 0
            if subset_name == "ocnli":
                if data_dict["label"] == "entailment":
                    label_id = 1
                elif data_dict["label"] == "contradiction":
                    label_id = 2
            elif subset_name == "rte":
                if data_dict["label"] == "entailment":
                    label_id = 0
                elif data_dict["label"] == "not_entailment":
                    label_id = 1

            data_dict["label"] = label_id

            outputs = prompt_template.apply(data_dict)

            prompt, target = None, None
            if len(outputs) >= 2:
                prompt = outputs[0]
                targets = outputs[1]
                target = targets[0].strip()

            example = prompt + prompt_target_sep + target
            examples.append(example)

    # print(f'examples: {examples}')
    # print(f'num_fewshot: {num_fewshot}')
    fewshot_context = example_sep.join(examples[:num_fewshot]) + example_sep

    return fewshot_context


def few_shot_play_file(
    fname,
    outfile,
    tokenizer,
    model,
    trial_num=5,
    steps=256,
    temperature=0.6,
    do_sample=True,
    top_k=5,
    top_p=None,
    until_n_eos=1,
    limit_samples=-1,
    dataset_name="",
    subset_name="",
    template_name="",
    num_fewshot=0,
    fewshot_file_path="",
):
    print(f"Generating by prompts from {fname}...")
    prompt_template = get_prompt_template(dataset_name, subset_name, template_name)

    if num_fewshot > 0:
        few_context = get_few_context(fewshot_file_path, prompt_template, num_fewshot, subset_name)
    else:
        few_context = ""

    count = 0
    with open(fname, "r") as f:
        with open(outfile, "w", encoding="utf-8") as fout:
            for line in f:
                count += 1
                if limit_samples > 0 and count >= limit_samples:
                    print(f"Reach limit_samples: {limit_samples}, stop.")
                    break
                try:
                    if subset_name == "ocnli":
                        data_dict = ocnli_processor(line)
                    elif subset_name == "rte":
                        data_dict = rte_processor(line)
                    elif dataset_name == "lambada":
                        data_dict = lambada_processor(line)

                    outputs = prompt_template.apply(data_dict)

                    prompt, target = None, None
                    if len(outputs) >= 2:
                        prompt = outputs[0]
                        targets = outputs[1]
                        target = targets[0].strip()

                    # add few_context
                    prompt = few_context + prompt
                    print("prompt: ", prompt)
                    print("target: ", target)

                    answer_choices_list = prompt_template.get_answer_choices_list(data_dict)
                    print("answer_choices_list: ", answer_choices_list)
                    target_idx = answer_choices_list.index(target)

                    prompt_tokens = tokenizer.tokenize(prompt)
                    # print('prompt_tokens: ', prompt_tokens)
                    input_ids = torch.tensor(tokenizer(prompt)["input_ids"]).unsqueeze(0).long()
                    prompt_ids = input_ids.tolist()[0]
                    prompt_ids_len = len(prompt_ids)

                    answer_choices_tokens = [
                        tokenizer.tokenize(answer_choice) for answer_choice in answer_choices_list
                    ]
                    answer_choices_ids = [
                        tokenizer(answer_choice)["input_ids"] for answer_choice in answer_choices_list
                    ]

                    for i in range(trial_num):
                        y = sample_generate(
                            model,
                            input_ids=input_ids.to(model.device),
                            steps=steps,
                            temperature=temperature,
                            do_sample=do_sample,
                            top_k=top_k,
                            top_p=top_p,
                            eos=tokenizer.eos_token_id,
                            until_n_eos=until_n_eos,
                        )
                        completion = tokenizer.convert_tokens_to_string_for_en(tokenizer.convert_ids_to_tokens(y))

                        input_mask = torch.ones_like(input_ids.to(model.device), device=model.device)
                        res = model.decode(
                            input_ids=input_ids.to(model.device),
                            input_mask=input_mask,
                        )
                        print(res["logits"].size())
                        json_str = json.dumps(
                            {
                                "prompt": f"{prompt}",
                                "prompt_tokens": f"{prompt_tokens}",
                                "prompt_ids": f"{prompt_ids}",
                                "prompt_ids_len": f"{prompt_ids_len}",
                                "answer_choices": f"{answer_choices_list}",
                                "answer_choices_tokens": f"{answer_choices_tokens}",
                                "answer_choices_ids": f"{answer_choices_ids}",
                                "target": f"{target}",
                                "target_idx": f"{target_idx}",
                                "label_name": f'{data_dict["label"]}',
                                "logits_by_answer_choice": f'{[res["logits"][0, prompt_ids_len - 1, ids[0]].tolist() for ids in answer_choices_ids]}',
                                "logits_size": f'{res["logits"].size()}',
                                "answer": f"{completion}",
                            },
                            ensure_ascii=False,
                        )

                        fout.write(json_str + "\n")
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
