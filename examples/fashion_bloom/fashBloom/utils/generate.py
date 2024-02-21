"""

将一些解码的过程抽象出来
Copied from FUXI, 参考标准类似：https://huggingface.co/spaces/THUDM/GLM-130B

"""
import json
import os
import sys
import tempfile
import traceback
from builtins import print

import pandas as pd
import torch
from cruise.utilities.hdfs_io import hcopy, hopen
from fashBloom.data.gpt.datamodule.load_utils import local_collator, local_dataset
from fashBloom.utils.processor import lambada_processor, ocnli_processor, rte_processor
from promptsource.templates import DatasetTemplates
from torch.nn import functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def sample(
    model_decode_func,
    x,
    steps,
    temperature=1.0,
    do_sample=False,
    top_k=None,
    top_p=None,
    eos=3,
    until_n_eos=1,
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
        raise ValueError("Either Top-K or Top-P, cannot chooes both")
    meet_eos_count = 0

    for k in range(steps):
        logits = model_decode_func(x)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # optionally crop probabilities to only the options whose prob larger than p
        elif top_p is not None:
            logits = top_p_logits(logits, top_p)
        probs = F.softmax(logits, dim=-1)

        # print(probs, probs.shape, 'probs')
        # sample from the distribution or take the most likely
        if do_sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        if ix.tolist()[0][0] == eos:  # 遇到eos 就停止生成
            meet_eos_count += 1
            if meet_eos_count >= until_n_eos:
                break
        if x is not None:
            x = torch.cat((x, ix), dim=1)
        else:
            x = ix
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
    # meet_eos_count = 0

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
    cumulative_probs = torch.cumsum(F.softmax(sorted_probs, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

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
    eos=5,
    until_n_eos=1,
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
        eos=eos,
        until_n_eos=until_n_eos,
    )
    return decoded_sentence.tolist()[0]


def sample_console(
    model_decode_func,
    x,
    steps,
    temperature=1.0,
    do_sample=False,
    top_k=None,
    top_p=None,
    eos=3,
    until_n_eos=1,
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
        raise ValueError("Either Top-K or Top-P, cannot chooes both")
    meet_eos_count = 0

    for k in range(steps):
        logits = model_decode_func(x)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # optionally crop probabilities to only the options whose prob larger than p
        elif top_p is not None:
            logits = top_p_logits(logits, top_p)
        probs = F.softmax(logits, dim=-1)

        # print(probs, probs.shape, 'probs')
        # sample from the distribution or take the most likely
        if do_sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        if ix.tolist()[0][0] == eos:  # 遇到eos 就停止生成
            meet_eos_count += 1
            if meet_eos_count >= until_n_eos:
                break
        if x is not None:
            x = torch.cat((x, ix), dim=1)
        else:
            x = ix
    return x


@torch.no_grad()
def sample_generate_console(
    model,
    input_ids,
    steps=32,
    temperature=1.0,
    do_sample=False,
    top_k=None,
    top_p=None,
    eos=5,
    until_n_eos=1,
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
    decoded_sentence = sample_console(
        model_decode_func=model_decode_func,
        x=input_ids,
        steps=steps,
        temperature=temperature,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        eos=eos,
        until_n_eos=until_n_eos,
    )
    return decoded_sentence.tolist()[0]


def play_console(
    tokenizer,
    model,
    trial_num=5,
    steps=256,
    temperature=0.6,
    do_sample=True,
    top_k=5,
    top_p=None,
    until_n_eos=1,
):
    print("Input your prompt here...")

    total_dialog = ""
    max_length = 256

    for line in sys.stdin:
        try:
            text = line.strip()
            if "/reset" in text:
                total_dialog = ""
                print("对话已清空")
                continue
            text = " user: " + text

            total_dialog = text
            # print('prompt: ', text)
            # print('tokens: ', tokenizer.tokenize(text))
            input_ids = tokenizer((total_dialog + " assistant: ").strip())["input_ids"]
            if len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]

            input_ids = torch.tensor(input_ids).unsqueeze(0).long()

            for i in range(trial_num):
                # y = sample_generate_console(model,
                #                     input_ids=input_ids.to(model.device),
                #                     steps=steps, temperature=temperature, do_sample=do_sample,
                #                     top_k=top_k,
                #                     top_p=top_p,
                #                     eos=tokenizer.eos_token_id,
                #                     until_n_eos=until_n_eos)
                # y = y.tolist()[0][1:]
                y = model.gpt.generate(
                    input_ids.to(model.device),
                    max_new_tokens=steps,
                    do_sample=True,
                    num_beams=1,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                completion = "".join(tokenizer.decode(y[0]))
                completion = completion[-(len(completion) - len(total_dialog)) :]
                completion.replace("##", "")
                print(f"{completion}")

                # total_dialog += 'assistant: ' + completion

            print("Input your prompt here...")
        except Exception as e:
            print(e)
            print(traceback.format_exc())


def play_file(
    fname,
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
):
    print(f"Generating by prompts from {fname}...")
    count = 0
    with hopen(fname) as f:
        for line in f:
            count += 1
            if limit_samples > 0 and count >= limit_samples:
                print(f"Reach limit_samples: {limit_samples}, stop.")
                break
            try:
                jl = json.loads(line)
                text = jl["page_info"]["core_content"].strip()
                print("prompt: ", text)
                print("tokens: ", tokenizer.tokenize(text))
                print("origin content", jl["page_info"]["core_content"][:steps])
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
                        eos=tokenizer.eos_token_id,
                        until_n_eos=until_n_eos,
                    )
                    # y = y.tolist()[0][1:]
                    completion = "".join(tokenizer.decode(y))
                    completion.replace("##", "")
                    print(f"[{i}]: {completion}")
            except Exception as e:
                print(e)
                print(traceback.format_exc())


def play_file_qa(
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
):
    print(f"Generating by prompts from {fname}...")
    # full_stop_input_ids = get_input_ids_of_stop_tokens(tokenizer)
    # full_stop_input_ids = None
    count = 0

    ori_outfile = None
    if outfile.startswith("hdfs"):
        ori_outfile = outfile
        outfile = os.path.join(tempfile.gettempdir(), os.path.basename(outfile))

    fname_tmp_path = None
    if fname.startswith("hdfs"):
        fname_tmp_path = os.path.join(tempfile.gettempdir(), os.path.basename(fname))
        hcopy(fname, fname_tmp_path)
    else:
        fname_tmp_path = fname

    val_df = pd.read_parquet(fname_tmp_path)

    if True:
        with open(outfile, "w", encoding="utf-8") as fout:
            # for line in f:
            for _, jl in val_df.iterrows():
                count += 1

                if count % 100 == 1:
                    print(f"[{count}/{len(val_df)}]...")

                if limit_samples > 0 and count >= limit_samples:
                    print(f"Reach limit_samples: {limit_samples}, stop.")
                    break
                try:
                    # jl = json.loads(line)
                    text = jl["question"].strip()
                    true_label = jl["answer"]
                    # print('prompt/query: ', text)
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
                            eos=tokenizer.eos_token_id,
                            until_n_eos=until_n_eos,
                        )

                        completion = "".join(tokenizer.decode(y))
                        completion.replace("##", "")

                        # print(f'[{i}]: {completion}')
                        id = jl["id"]
                        sent1 = jl["sent1"]

                        json_str = json.dumps(
                            {
                                "id": f"{id}",
                                "sent1": f"{sent1}",
                                "question": f"{text}",
                                "ground truth answer": f"{true_label}",
                                "output": f"[{i}]: {completion}",
                            },
                            ensure_ascii=False,
                        )

                        if count % 2 == 1:
                            print(f"{json_str}")

                        fout.write(json_str + "\n")

                except Exception as e:
                    print(e)
                    print(traceback.format_exc())

        if ori_outfile is not None:
            os.system(f"hdfs dfs -put -f {outfile} {ori_outfile}")


@torch.no_grad()
def batch_sample(
    model_decode_func,
    x,
    attention_mask,
    steps,
    temperature=1.0,
    do_sample=False,
    top_k=None,
    top_p=None,
    eos=3,
    until_n_eos=1,
):
    """
    Take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time.
    """
    if top_k is not None and top_p is not None:
        raise ValueError("Either Top-K or Top-P, cannot chooes both")
    meet_eos_count = 0

    check_eos_list = [-1] * x.size(0)
    # print('check_eos_list', len(check_eos_list))

    for k in range(steps):
        logits = model_decode_func(x, attention_mask)

        # print('x size', x.size())
        # print('logits size', logits.size())
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # optionally crop probabilities to only the options whose prob larger than p
        elif top_p is not None:
            logits = top_p_logits(logits, top_p)
        probs = F.softmax(logits, dim=-1)

        # print(probs, probs.shape, 'probs')
        # sample from the distribution or take the most likely
        if do_sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue

        ix_list = ix.tolist()

        # print('ix list is', ix.tolist())
        for j in range(x.size(0)):
            if check_eos_list[j] == -1 and ix_list[j][0] == eos:
                check_eos_list[j] = x.size(1)
                meet_eos_count += 1

        if meet_eos_count >= x.size(0):
            break

        if x is not None:
            x = torch.cat((x, ix), dim=1)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones_like(ix, device=attention_mask.device),
                ),
                dim=1,
            )
        else:
            x = ix
            attention_mask = torch.ones_like(x, device=x.device)

        # print('='*50)

    return x, check_eos_list


@torch.no_grad()
def batch_sample_generate(
    model,
    input_ids,
    attention_mask,
    steps=32,
    temperature=1.0,
    do_sample=False,
    top_k=None,
    top_p=None,
    eos=5,
    until_n_eos=1,
    seq_len=None,
):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """

    # 2. 构造 模型解码的func wrapper
    def model_decode_func(x, mask):
        # input_mask = torch.ones_like(x, device=x.device)
        res = model.decode(input_ids=x, input_mask=mask)
        return res["logits"]

    # 3. 生成结果
    decoded_sentence, check_eos_list = batch_sample(
        model_decode_func=model_decode_func,
        x=input_ids,
        attention_mask=attention_mask,
        steps=steps,
        temperature=temperature,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        eos=eos,
        until_n_eos=until_n_eos,
    )

    decoded_sentence = decoded_sentence.tolist()
    max_len = max(seq_len)
    # print('check_eos_list', check_eos_list)
    # print('seq len', seq_len)
    for i in range(len(decoded_sentence)):
        # print(decoded_sentence[i])
        if check_eos_list[i] != -1:
            decoded_sentence[i] = decoded_sentence[i][max_len - seq_len[i] : check_eos_list[i]]
        else:
            decoded_sentence[i] = decoded_sentence[i][max_len - seq_len[i] :]
        # print(decoded_sentence[i])
        # print('\n')

    return decoded_sentence


def play_file_qa_batch(
    fname,
    outfile,
    tokenizer,
    model,
    trial_num=1,
    val_bsz=4,
    steps=256,
    temperature=0.6,
    do_sample=True,
    top_k=5,
    top_p=None,
    until_n_eos=1,
    limit_samples=-1,
):
    print(f"Generating by prompts from {fname}...")
    count = 0
    ori_outfile = None
    if outfile.startswith("hdfs"):
        ori_outfile = outfile
        outfile = os.path.join(tempfile.gettempdir(), os.path.basename(outfile))

    fname_tmp_path = None
    if fname.startswith("hdfs"):
        fname_tmp_path = os.path.join(tempfile.gettempdir(), os.path.basename(fname))
        hcopy(fname, fname_tmp_path)
    else:
        fname_tmp_path = fname

    collator = local_collator(pad_token_id=tokenizer.pad_token_id)
    test_dataset = local_dataset(fname_tmp_path, tokenizer, data_type="parquet", shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=val_bsz, shuffle=False, collate_fn=collator)

    dataset_len = len(test_dataset)

    with open(outfile, "w", encoding="utf-8") as fout:
        for examples in test_loader:
            seq_len = examples["seq_len"]
            model_inputs = examples["padded_inputs"]
            # print(model_inputs)

            model_inputs = {k: torch.tensor(v).long().to(model.device) for k, v in model_inputs.items()}

            # print(model_inputs['input_ids'])
            # print(examples)
            try:
                for i in range(trial_num):
                    ys = batch_sample_generate(
                        model,
                        input_ids=model_inputs["input_ids"].to(model.device),
                        attention_mask=model_inputs["attention_mask"].to(model.device),
                        steps=steps,
                        temperature=temperature,
                        do_sample=do_sample,
                        top_k=top_k,
                        top_p=top_p,
                        eos=tokenizer.eos_token_id,
                        until_n_eos=until_n_eos,
                        seq_len=seq_len,
                    )

                    for idx in range(len(ys)):
                        y = ys[idx]
                        completion = "".join(tokenizer.decode(y))
                        # print(completion)
                        completion.replace("##", "")

                        json_str = json.dumps(
                            {
                                "id": f"{examples['id'][idx]}",
                                "country": f"{examples['country'][idx]}",
                                "sent1": f"{examples['sent1'][idx]}",
                                "question": f"{examples['question'][idx]}",
                                "ground truth answer": f"{examples['answer'][idx]}",
                                "output": f"[{i}]: {completion}",
                            },
                            ensure_ascii=False,
                        )

                        fout.write(json_str + "\n")

                        count += 1
                        if count % 100 == 1:
                            print(f"[{count}/{dataset_len}]...")
                        if count % 100 == 1:
                            # pass
                            print(f"{json_str}")

            except Exception as e:
                print(e)
                print(traceback.format_exc())

            # break

    if ori_outfile is not None:
        os.system(f"hdfs dfs -put -f {outfile} {ori_outfile}")

    return


def play_file_qa_batch_from_offical_generate(
    fname,
    outfile,
    tokenizer,
    model,
    trial_num=1,
    val_bsz=4,
    steps=256,
    temperature=0.6,
    do_sample=True,
    top_k=5,
    top_p=None,
    until_n_eos=1,
    limit_samples=-1,
):
    print(f"Generating by prompts from {fname}...")
    count = 0
    ori_outfile = None
    if outfile.startswith("hdfs"):
        ori_outfile = outfile
        outfile = os.path.join(tempfile.gettempdir(), os.path.basename(outfile))

    fname_tmp_path = None
    if fname.startswith("hdfs"):
        fname_tmp_path = os.path.join(tempfile.gettempdir(), os.path.basename(fname))
        hcopy(fname, fname_tmp_path)
    else:
        fname_tmp_path = fname

    collator = local_collator(pad_token_id=tokenizer.pad_token_id)
    test_dataset = local_dataset(fname_tmp_path, tokenizer, data_type="parquet", shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=val_bsz, shuffle=False, collate_fn=collator)

    dataset_len = len(test_dataset)

    with open(outfile, "w", encoding="utf-8") as fout:
        for examples in test_loader:
            # seq_len = examples["seq_len"]
            model_inputs = examples["padded_inputs"]
            # print(model_inputs)

            model_inputs = {k: torch.tensor(v).long().to(model.device) for k, v in model_inputs.items()}

            # print(model_inputs['input_ids'])
            # print(examples)
            try:
                for i in range(trial_num):
                    ys = model.gpt.generate(
                        model_inputs["input_ids"].to(model.device),
                        attention_mask=model_inputs["attention_mask"].to(model.device),
                        max_new_tokens=steps,
                        do_sample=True,
                        num_beams=1,
                        temperature=temperature,
                        top_p=0.95,
                        top_k=50,
                        repetition_penalty=1.3,
                        pad_token_id=tokenizer.pad_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    ).tolist()

                    for idx in range(len(ys)):
                        y = ys[idx]
                        """
                        这里把前面的输入question截断，因为会有很多<pad>
                        from: len(model_inputs['input_ids'][idx])
                        to: 第一个tokenizer.eos_token_id
                        """
                        y = y[len(model_inputs["input_ids"][idx]) :]
                        y = y[: y.index(tokenizer.eos_token_id)] if tokenizer.eos_token_id in y else y

                        completion = "".join(tokenizer.decode(y))
                        # print(completion)
                        completion.replace("##", "")

                        json_str = json.dumps(
                            {
                                "id": f"{examples['id'][idx]}",
                                "sent1": f"{examples['sent1'][idx]}",
                                "question": f"{examples['question'][idx]}",
                                "ground truth answer": f"{examples['answer'][idx]}",
                                "output": f"[{i}]: {completion}",
                            },
                            ensure_ascii=False,
                        )

                        fout.write(json_str + "\n")

                        count += 1
                        if count % 100 == 1:
                            print(f"[{count}/{dataset_len}]...")
                        if count % 100 == 1:
                            # pass
                            print(f"{json_str}")

            except Exception as e:
                print(e)
                print(traceback.format_exc())

    if ori_outfile is not None:
        os.system(f"hdfs dfs -put -f {outfile} {ori_outfile}")

    return


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
                                "logits_by_answer_choice": f'{[res["logits"][0,prompt_ids_len-1, ids[0]].tolist() for ids in answer_choices_ids]}',  # noqa: E501
                                "logits_size": f'{res["logits"].size()}',
                                "answer": f"{completion}",
                            },
                            ensure_ascii=False,
                        )

                        fout.write(json_str + "\n")
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
