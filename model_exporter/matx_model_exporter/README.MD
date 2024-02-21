## EasyGuard预训练模型上线指南

> 本仓库为主要内容为将EasyGuard预训练后的模型转换为laplace(matx)服务，并进行benard部署
> 主要包含三个步骤：将`xxx.pt`文件转换为`xxx.jit`文件， 将`xxx.jit`转换为matx服务，将
> 对应的matx上线benard

## 1.文件说明

| 文件名                      |功能|
|--------------------------|---|
| `convert_model_to_jit.py` |将`xxx.pt`文件转化为`xxx.jit`文件。如何trace可以参考pytorch官方文档|
| `convert_jit_to_matx.py` |将`xxx.jit`文件转化为`matx`服务资源文件。如何trace可以参考pytorch官方文档|
| `service_validation.py`  |本地服务验证|

## 2.参考文档
- NLP模型训练trace部署指南：https://bytedance.feishu.cn/docx/doxcnzKN98cM1B5Gr5yKRmpa31e
- 代码示例：https://code.byted.org/ecom_govern/nlp_model_tianke
- Laplace&Ferminco推理：https://bytedance.feishu.cn/wiki/wikcnGlaLxBw9Lq2YDtrjbNqIhf#