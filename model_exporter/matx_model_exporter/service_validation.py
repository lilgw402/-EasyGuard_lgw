# -*- coding: utf-8 -*-
from laplace import Laplace

laplace = Laplace("127.0.0.1:9898")
# if service is on Bernard: laplace = Laplace('aml.nlp.ljh_hello_world')
input_lists = {"text": ["这个衣服掉色严重".encode()]}

model_name = "matx_service_for_clothes"
rsp = laplace.matx_inference(model_name, input_lists)
print(rsp)
