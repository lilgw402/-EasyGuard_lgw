# 简介

本目录介绍了 `海外CCR` 场景下两种加速推理的方式，分别是基于 `vLLM` 和基于 `量化+Xperf`，使用的模型是Llama2-7B模型。  
其中，`handler_hf.py` 使用了 `transformers` 原生的 `generate` 接口。调用了 `transformers` 默认的一些加速方法，比如 `float16`、`use_cache` 等。  
`handler_vllm.py` 使用了 `vLLM` 的加速方式，主要优势是提升了模型的吞吐量，要求 `batch size` 非常大。
`handler_quant_xperf.py` 使用了公司内部开发的 `xperf` 加速工具，同时 `xperf` 支持输入量化后的模型。不论是模型的吞吐量还是延时，都有明显的正向收益。


# 使用方式
```python3
export PYTHONPATH=`pwd`:${PYTHONPATH}
cd examples/inference/laplace_python3/ccr_oversea/
pip3 install -r requirements.txt
# 下载模型和数据集
python3 tos_helper.py 
# 测试transformers方式的推理耗时
python3 handler_hf.py
# 测试vLLM方式的推理耗时
python3 handler_vllm.py
# 测试量化+xperf的推理耗时
python3 handler_vllm.py
```

# 精度验证方式

```python3
python3 evaluation.py [待评测文件的路径] llama_7b_ccr/product_report_test_8k.parquet
```