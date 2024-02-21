# GandalfOnCruise
  基于cruise和Easyguard预训练模型能力的新版Gandalf指北详见：
  [直播甘道夫NN模型训练部署指北Cruise版本](https://bytedance.feishu.cn/docx/Fh4udJgZQofrY3xXGXycaOA6n5d)
# 一、Gandalf为什么要使用Cruise
1. Cruise针对原生torch在接入公司内部算力(Arnold)，内部数据(kv,parquet)，弹性训练(Elastic Training)，分布式训练(DDP,DDPNPU,DeepSpeed)等方面做了定制优化，让算法同学专注于业务算法开发，减少了其他wet-hands工作的开发量(读取远程数据/针对分布式训练手写梯度更新等)。
2. 组内内容理解基建所有成果均托管于EasyGuard，包括多模态预训练，NLP预训练，LLM能力探索等。而EasyGuard的模型能力目前是基于纯粹的Cruise框架训练的，使用Cruise开发Gandalf模型可以快速复用组内现有基建能力
总上所述，将现在托管在其他框架的Gandalf模型整体流程适配Cruise，打通了底层内容能力理解和上层业务应用之间的藩篱，并托管在[EasyGuard/gandalf](https://code.byted.org/ecom_govern/EasyGuard/tree/gandalf/examples/gandalf)。
3. 
# 二、如何在Cruise上训练Gandalf模型
参考[直播甘道夫NN模型训练部署指北Cruise版本](https://bytedance.feishu.cn/docx/Fh4udJgZQofrY3xXGXycaOA6n5d)的第四部分