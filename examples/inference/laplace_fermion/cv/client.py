import time

import euler
from idl.base.base_thrift import *  # noqa: F403, F401
from idl.fermion_thrift import DataType, FermionCore, InferRequest, Tensor, TensorSet


def InferImage(image_file_paths, client):
    req = InferRequest()
    req_input = []

    binary_string = []
    for image_file_path in image_file_paths:
        with open(image_file_path, mode="rb") as file:  # b is important -> binary
            file_content = file.read()
            binary_string.append(file_content)

    for s in binary_string:
        t = Tensor()
        t.dtype = DataType.STRING
        t.shape = []
        t.str_data = [s]
        req_input.append(TensorSet(tensors={"image": t}))

    req.input = req_input

    t = time.time()
    print("===========Detect")
    resp = client.Infer(req)
    print("===========Infer done, time usage: %s" % (time.time() - t))
    outputs = resp.output
    print(resp)
    for output in outputs:
        output = output.tensors["_3"].float_data  # 这里"_3"和全图导出里面的代码是对应的
        print(output)


if __name__ == "__main__":
    img_list = ["demo.jpg"]
    # P.S.M和hl换成自己的即可
    client = euler.Client(FermionCore, "sd://ecom.govern.swin_model_20231211?idc=hl&cluster=default", timeout=1)
    InferImage(img_list, client)
