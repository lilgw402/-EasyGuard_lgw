# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-02-27 20:36:34
# Modified: 2023-02-27 20:36:34
from typing import Optional

from addict import Dict
from dataset.template_datasets.TemplateCruiseDataModule import TemplateCruiseDataModule, TemplateParquetFeatureProvider
from utils.registry import DATASETS, FEATURE_PROVIDERS


@FEATURE_PROVIDERS.register_module()
class GandalfParquetFeatureProvider(TemplateParquetFeatureProvider):
    def __init__(self):
        super(GandalfParquetFeatureProvider, self).__init__()
        pass

    def process_feature_dense(self, features):
        pass

    def process_text(self, text):
        # 加载预处理参数
        pass

    def process_image(self, image):
        pass

    def process(self, data):
        pass


@DATASETS.register_module()
class GandalfParquetCruiseDataModule(TemplateCruiseDataModule):
    def __init__(self):
        super(GandalfParquetCruiseDataModule, self).__init__()
        self.save_hparams()

    def setup(self, stage: Optional[str] = None) -> None:
        # print(self.hparams)
        # print('self.hparams.dataset',type(self.hparams.dataset),self.hparams.dataset)
        self.dataset = Dict(self.hparams.dataset)
        self.feature_provider = Dict(self.hparams.feature_provider)
        self.data_factory = Dict(self.hparams.data_factory)
        self.total_cfg = Dict(
            {
                "dataset": self.dataset,
                "feature_provider": self.feature_provider,
                "data_factory": self.data_factory,
            }
        )
