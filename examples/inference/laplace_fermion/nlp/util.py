import base64
import json
import logging
from enum import IntEnum
from io import BytesIO

import fermion_ops
import numpy as np
import torch


def expand_service_input(s_ins):
    """
    Expand your service_input with bs-dim as it will be done inside FermionCore server.

    Tensor w/ shape (x, y, ..., n)  ->  (1, x, y, ..., n)
    str                             ->  List[str]
    List[str]                       ->  List[List[str]]
    """
    bs_s_ins = []
    for s_in in s_ins:
        if isinstance(s_in, torch.Tensor):
            bs_s_ins.append(torch.unsqueeze(s_in, 0))
        else:
            bs_s_ins.append([s_in])
    return tuple(bs_s_ins)


def parse_to_service_output(m_outs):
    """
    Squeeze the bs-dim of your model outputs as it will be done inside FermionCore server.

    Tensor w/ shape (1, x, y, ..., n)  ->  (x, y, ..., n)
    List[str]                          ->  str (at index 0)
    """
    s_outs = []
    for m_out in m_outs:
        if isinstance(m_out, torch.Tensor):
            s_outs.append(torch.squeeze(m_out, 0))
        else:
            s_outs.append(m_out[0])
    return tuple(s_outs)


class Datatype(IntEnum):
    INVALID = (0,)
    INT8 = (8,)
    INT16 = (16,)
    INT32 = (32,)
    INT64 = (64,)
    UINT8 = (108,)
    UINT16 = (116,)
    UINT32 = (132,)
    UINT64 = (164,)
    FLOAT16 = (216,)
    FLOAT32 = (232,)
    FLOAT64 = (264,)
    STRING = (300,)


TensorTypeMapping = {
    torch.float32: Datatype.FLOAT32,
    torch.float: Datatype.FLOAT32,
    torch.float64: Datatype.FLOAT64,
    torch.double: Datatype.FLOAT64,
    torch.float16: Datatype.FLOAT16,
    torch.half: Datatype.FLOAT16,
    torch.uint8: Datatype.UINT8,
    torch.int8: Datatype.INT8,
    torch.int16: Datatype.INT16,
    torch.short: Datatype.INT16,
    torch.int32: Datatype.INT32,
    torch.int: Datatype.INT32,
    torch.int64: Datatype.INT64,
    torch.long: Datatype.INT64,
}


class TensorInfo:
    dtype = Datatype.INVALID
    name = ""
    min_shape = []
    max_shape = []
    modal_type = ""

    def to_dict(self):
        return {
            "dtype": self.dtype,
            "min_shape": list(self.min_shape),
            "max_shape": list(self.max_shape),
            "name": self.name,
            "modal_type": self.modal_type,
        }


def modal_type_for_string_input(str_input):
    abbrv_in = str_input[:5]
    try:
        logging.debug(f"Try treating string input '{abbrv_in}' as image binary...")
        decode_op = fermion_ops.ImageDecodeOp()
        decode_op([str_input], is_video=False)
        logging.debug(f"Image decode successful. Marking string input '{abbrv_in}' as IMAGE_DATA.")
        return "IMAGE_DATA"
    except:  # noqa: E722
        pass

    try:
        logging.debug(f"Try treating string input '{abbrv_in}' as audio binary...")
        audio_op = fermion_ops.AudioDecodeOp()
        audio_op(str_input)
        logging.debug(f"Audio decode successful. Marking string input '{abbrv_in}' as AUDIO_DATA.")
        return "AUDIO_DATA"
    except:  # noqa: E722
        pass

    logging.debug(f"Marking string input '{abbrv_in}' as TEXT.")
    return "TEXT"


def modify_tensor_info(tensor, tensor_name, tensor_info):
    tensor_info.name = tensor_name
    if isinstance(tensor, torch.Tensor):
        if tensor_info.min_shape:
            assert len(tensor.shape) == len(tensor_info.min_shape)
            assert len(tensor.shape) == len(tensor_info.max_shape)
            for j in range(len(tensor.shape)):
                if tensor.shape[j] < tensor_info.min_shape[j]:
                    tensor_info.min_shape[j] = tensor.shape[j]

                if tensor.shape[j] > tensor_info.max_shape[j]:
                    tensor_info.max_shape[j] = tensor.shape[j]
        else:
            tensor_info.min_shape = list(tensor.shape)
            tensor_info.max_shape = list(tensor.shape)
        tensor_info.dtype = TensorTypeMapping[tensor.dtype]
        tensor_info.modal_type = "FEATURE"
    elif isinstance(tensor, list) and all(
        isinstance(element, str) or isinstance(element, bytes) for element in tensor
    ):
        tensor_info.min_shape = [1]
        tensor_info.max_shape = [1]
        tensor_info.dtype = Datatype.STRING
        new_modal_type = modal_type_for_string_input(tensor[0])
        if not tensor_info.modal_type:
            tensor_info.modal_type = new_modal_type
        if tensor_info.modal_type != new_modal_type:
            logging.warning(
                "Modal type different, previous is:" + tensor_info.modal_type + "  new is:" + new_modal_type
            )
    elif isinstance(tensor, list) and isinstance(tensor[0], list):
        if tensor_info.min_shape:
            if len(tensor[0]) < tensor_info.min_shape[1]:
                tensor_info.min_shape[1] = len(tensor[0])
            if len(tensor[0]) > tensor_info.max_shape[1]:
                tensor_info.max_shape[1] = len(tensor[0])
        else:
            tensor_info.min_shape = [1, len(tensor[0])]
            tensor_info.max_shape = [1, len(tensor[0])]
        tensor_info.dtype = Datatype.STRING
        new_modal_type = modal_type_for_string_input(tensor[0][0])
        if not tensor_info.modal_type:
            tensor_info.modal_type = new_modal_type
        if tensor_info.modal_type != new_modal_type:
            logging.warning(
                "Modal type different, previous is:" + tensor_info.modal_type + "  new is:" + new_modal_type
            )
    else:
        logging.error(f"Export With Error, not support tensor type in input: {tensor}")
        return False
    return True


def save(model, data, save_dir):
    script_module = torch.jit.script(model)

    """
    Inspect the forward code and get the calling relationship
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(model.forward)))
    nodes = ast.walk(tree)
    service_input_name = []
    service_output_name = []
    pre_input_name = []
    pre_output_name = []
    infer_input_name = []
    infer_output_name = []
    post_input_name = []
    post_output_name = []

    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args:
                if arg.arg != "self":
                    service_input_name.append(arg.arg)
        if isinstance(node, ast.Return):
            if not isinstance(node.value, ast.Tuple):
                logging.info("The return type of forward function must be one Tuple")
                return False

            for id_ in node.value.elts:
                service_output_name.append(id_.id)
        if isinstance(node, ast.Assign):
            func = node.value.func
            args = node.value.args
            rets = node.targets
            output_tensor_name = []
            input_tensor_name = []
            if len(rets) != 1 or (not isinstance(rets[0], ast.Tuple)):
                logging.info("The return type of pre/infer/post function must be one Tuple")
                return False
            for id in rets[0].elts:
                output_tensor_name.append(id.id)
            for arg in args:
                while isinstance(arg, ast.Call):
                    arg = arg.func
                if isinstance(arg, ast.Attribute):
                    arg = arg.value
                input_tensor_name.append(arg.id)
            if func.attr == "preprocess_single":
                pre_input_name.extend(input_tensor_name)
                pre_output_name.extend(output_tensor_name)
            elif func.attr == "infer_batched":
                infer_input_name.extend(input_tensor_name)
                infer_output_name.extend(output_tensor_name)
            elif func.attr == "postprocess_single":
                post_input_name.extend(input_tensor_name)
                post_output_name.extend(output_tensor_name)
            else:
                logging.info("you should not have other function call inside the forward")
                return False

    def _gen_empty_info(length: int):
        return [TensorInfo() for _ in range(length)]

    service_in_info = _gen_empty_info(len(service_input_name))
    pre_in_info = _gen_empty_info(len(pre_input_name))
    pre_out_info = _gen_empty_info(len(pre_output_name))
    infer_in_info = _gen_empty_info(len(infer_input_name))
    infer_out_info = _gen_empty_info(len(infer_output_name))
    post_in_info = _gen_empty_info(len(post_input_name))
    post_out_info = _gen_empty_info(len(post_output_name))
    service_out_info = _gen_empty_info(len(service_output_name))

    sample_data_dict = {}
    name_list = (
        service_input_name
        + pre_input_name
        + pre_output_name
        + infer_input_name
        + infer_output_name
        + post_input_name
        + post_output_name
        + service_output_name
    )
    for i in name_list:
        sample_data_dict[i] = []

    ts_dict = {}
    for ts in data:
        ts = expand_service_input(ts)

        # Update data dict
        for i in range(len(ts)):
            ts_dict[service_input_name[i]] = ts[i]
            sample_data_dict[service_input_name[i]].append(ts[i])

        # Update service_in_info
        for i in range(len(service_in_info)):
            if not modify_tensor_info(ts[i], service_input_name[i], service_in_info[i]):
                return False

        # Update pre_in_info
        for i in range(len(pre_in_info)):
            if not modify_tensor_info(ts_dict[pre_input_name[i]], pre_input_name[i], pre_in_info[i]):
                return False

        # Call preprocess
        preprocess_single_input_list = []
        for i in pre_input_name:
            preprocess_single_input_list.append(ts_dict[i])
        logging.info("Try calling preprocess_single")
        if hasattr(model, "preprocess_single"):
            ts = script_module.preprocess_single(*(*preprocess_single_input_list,))
            for i in range(len(ts)):
                ts_dict[pre_output_name[i]] = ts[i]
                sample_data_dict[pre_output_name[i]].append(ts[i])
            # Update pre_out_info
            for i in range(len(pre_out_info)):
                if not modify_tensor_info(ts_dict[pre_output_name[i]], pre_output_name[i], pre_out_info[i]):
                    return False
        else:
            logging.debug("Model has no preprocess_single, skipping...")
        # Update infer_in_info
        for i in range(len(infer_in_info)):
            if not modify_tensor_info(ts_dict[infer_input_name[i]], infer_input_name[i], infer_in_info[i]):
                return False
        # infer
        infer_input_list = []
        for i in infer_input_name:
            infer_input_list.append(ts_dict[i].cuda())
        logging.debug("Try calling infer_batched")
        ts = script_module.infer_batched(*(*infer_input_list,))
        for i in range(len(ts)):
            ts_dict[infer_output_name[i]] = ts[i]
            sample_data_dict[infer_output_name[i]].append(ts[i])
        # Update infer_out_info
        for i in range(len(infer_out_info)):
            if not modify_tensor_info(ts_dict[infer_output_name[i]], infer_output_name[i], infer_out_info[i]):
                return False
        # Update post_in_info
        for i in range(len(post_in_info)):
            if not modify_tensor_info(ts_dict[post_input_name[i]], post_input_name[i], post_in_info[i]):
                return False
        # Post Process
        post_input_list = []
        for i in post_input_name:
            post_input_list.append(ts_dict[i].cpu())
        logging.debug("Try calling postprocess_single")
        if hasattr(model, "postprocess_single"):
            ts = script_module.postprocess_single(*(*post_input_list,))
            for i in range(len(ts)):
                ts_dict[post_output_name[i]] = ts[i]
                sample_data_dict[post_output_name[i]].append(ts[i])
            for i in range(len(post_out_info)):
                if not modify_tensor_info(ts_dict[post_output_name[i]], post_output_name[i], post_out_info[i]):
                    return False
        else:
            logging.debug("Model has no postprocess_single, skipping...")

        # Update service_out_info
        for i in range(len(service_out_info)):
            if not modify_tensor_info(ts_dict[service_output_name[i]], service_output_name[i], service_out_info[i]):
                return False

    # Use Numpy to save the sample data
    for sample_data in sample_data_dict.keys():
        dt = sample_data_dict[sample_data]
        for i in range(len(dt)):
            if isinstance(dt[i], torch.Tensor):
                np_array = dt[i].cpu().detach().numpy()
            elif isinstance(dt[i], list):
                np_array = np.array(dt[i])
            else:
                logging.error("Export With Error, not support tensor type in input")
                return False
            dt[i] = np_array
        sample_data_dict[sample_data] = []
        for one_sample in dt:
            memfile = BytesIO()
            np.save(memfile, one_sample)
            memfile.seek(0)
            save_dt = base64.b64encode(memfile.read()).decode("ascii")
            sample_data_dict[sample_data].append(save_dt)

    import os

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    service_info_dict = {
        "service_in": [x.to_dict() for x in service_in_info],
        "pre_in": [x.to_dict() for x in pre_in_info],
        "pre_out": [x.to_dict() for x in pre_out_info],
        "infer_in": [x.to_dict() for x in infer_in_info],
        "infer_out": [x.to_dict() for x in infer_out_info],
        "post_in": [x.to_dict() for x in post_in_info],
        "post_out": [x.to_dict() for x in post_out_info],
        "service_out": [x.to_dict() for x in service_out_info],
    }
    logging.info("Model Info:")
    logging.info(service_info_dict)

    extra_files = {"service_info": json.dumps(service_info_dict), "sample_data": json.dumps(sample_data_dict)}
    script_module.save(os.path.join(save_dir, "model.pt"), _extra_files=extra_files)
    logging.debug("Export Successfully")
    return True


def forward(model, service_input):
    logging.info("Forwarding sample service input...")
    script_model = torch.jit.script(model)
    outputs = []
    for i, s_in in enumerate(service_input):
        in_ = expand_service_input(s_in)
        out_ = script_model.forward(*in_)
        outputs.append(parse_to_service_output(out_))
        logging.info(f"Done forward service_input #{i}")
    return outputs
