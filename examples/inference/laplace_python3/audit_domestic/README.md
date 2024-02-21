# 简介
本目录介绍了国内审核任务场景下基于 `量化+xperf` 的模型推理加速方式。模型使用的是 `alpaca-2-13B` 模型。  
其中:  
`handler.py` 使用了 `transformers` 原生的 `generate` 接口；  
`handler_xperf.py` 使用了 `xperf` 的加速方式；  
`handler_quant_xperf.py` 使用了 `量化+xperf` 的加速方式；

# 使用方式
```python
export PYTHONPATH=`pwd`:${PYTHONPATH}
cd examples/inference/laplace_python3/audit_domestic/
# 下载模型和数据集(要保证开发机的内存空间比模型的占用空间要大)
python3 tos_helper.py 
# 测试transformers方式的推理耗时
python3 handler.py
# 测试基于xperf的加速方式的推理耗时
python3 handler_xperf.py 
# 测试基于量化+xperf的加速方式的推理耗时
python3 handler_quant_xperf.py 
```

# 运行验证集
```python
python3 validation.py
```

# 部署到Bernard
可以参考[文档](https://bytedance.feishu.cn/docx/RBS9du3WCoRTCExcZkccBrDxnqc)