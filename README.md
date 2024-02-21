# EasyGuard

(Pre-Trained Models and Applications for E-com. Govern.)

[Documentation]() |
[Contributors](https://code.byted.org/ecom_govern/EasyGuard/) |
[Release Notes]()

EasyGuard是基于AML的Cruise框架扩展的业务算法的代码库，旨在给平台治理提供NLP，CV，和多模态的基础算法和解决方案。目前框架支持治理侧的预训练模型库FashionModdels，知识蒸馏，多任务学习，和模型推理加速等。

EasyGuard的使用说明详见：

* [EasyGuard的框架设计](https://bytedance.feishu.cn/docx/doxcnjT9CWaIH1PNDMg5TLUz7UN)
* [内容理解基建开发部署流程入门](https://bytedance.feishu.cn/docx/doxcn87hZjRkmyC2lWjkcQiyghh)
* [EasyGuard / CRUISE开发入门](https://bytedance.feishu.cn/wiki/wikcnFqJR5Y5dgswiuN4Vs5yy6e)
* [电商治理预训练模型库](https://bytedance.feishu.cn/sheets/shtcnJU6aAYhLP1wdYXFyPXH7mc)


[Warning:代码处于快速迭代期，核心代码需要UT保护，不然可能会被破坏!]()

[Warning:代码库branches比较多整体比较大，建议只clone master分支：git clone -b master --single-branch https://code.byted.org/ecom_govern/EasyGuard.git]()

## Job
推荐用Merlin
* Merlin中国区（MLX-Lab）: https://ml.bytedance.net/development/repos/207/detail?sid=3c70c89545eb65c3
* Merlin海外：https://ml.byteintl.net/development/repos/22/detail?sid=fbed5f4d85cdce20

# 电商场景预训练算法服务
* 文档详见：[预训练算法服务](https://bytedance.feishu.cn/wiki/wikcnrmcpmz5RAB89yJhWd0jFZg)

* FashionModel模型列表，详见[电商治理预训练模型库](https://bytedance.feishu.cn/sheets/shtcnJU6aAYhLP1wdYXFyPXH7mc)

* EasyGuard示例，详见[EasyGuard Examples](https://code.byted.org/ecom_govern/EasyGuard/tree/master/examples)

* 调用方式详见[电商治理模型FashionModels使用文档](https://bytedance.feishu.cn/wiki/wikcnBlgTsEuyDo1ZtYXW38k4Gf)

* 调用示例

调用模型：

```python
archive = "deberta_base_6l"
# 读取tokenizer
my_tokenizer = AutoTokenizer.from_pretrained(archive)
# 读取model
my_model = AutoModel.from_pretrained(archive)
```

查看模型：

```python
from easyguard.utils import list_pretrained_models

list_pretrained_models()
```

## 问题反馈

* 直接提[issues](https://code.byted.org/ecom_govern/EasyGuard/issues)

* 记录到文档里：[EasyGuard问题反馈](https://bytedance.feishu.cn/docx/Hk8NdiLkWofEzUxJGn4cot9KnwP)


## 依赖

EasyGuard依赖公司内外的NLP，CV，和多模态的框架来构建基础业务算法

* [Cruise框架](https://codebase.byted.org/repo/data/cruise)
* [Huggingface/Transformers框架](https://github.com/huggingface/transformers)


## CopyRight
2021 ByteDance Inc. All Rights Reserved.
