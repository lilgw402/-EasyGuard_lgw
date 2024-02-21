import time

import euler
from idl.base.base_thrift import *  # noqa: F403, F401
from idl.fermion_thrift import DataType, FermionCore, InferRequest, Tensor, TensorSet


def InferTexts(texts, client):
    req = InferRequest()
    req_input = []

    for text in texts:
        txt = Tensor()
        txt.dtype = DataType.STRING
        txt.shape = []  # 注意shape为空list
        txt.str_data = [text]
        req_input.append(TensorSet(tensors={"text": txt}))

    req.input = req_input
    print("===========Infer")
    start = time.time()
    resp = client.Infer(req)
    print("===========Infer done, time usage: %s" % (time.time() - start))
    # print(resp)
    # print(resp.output)
    for output in resp.output:
        output = output.tensors["ret"].float_data
        print(output)
        print("=" * 100)


if __name__ == "__main__":
    texts = [
        "码数偏大,比想象中的大很多.",
    ]
    # P.S.M和集群换成自己的
    client = euler.Client(FermionCore, "sd://ecom.govern.ccr_order_torch1131_cu117?idc=yg&cluster=default", timeout=30)
    InferTexts(texts, client)
