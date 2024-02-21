import torch
from utils.driver import get_logger
from utils.registry import OPTIMIZERS


class BaseOptimizer(object):
    def __init__(self, models, lr, wd, wd_ignore_bias=False, **kwargs):
        self.models = models
        self.lr = lr
        self.wd = wd
        self.wd_ignore_bias = wd_ignore_bias
        self.params = list()
        self.decay_params = list()
        self.no_decay_params = list()
        self.custom_keys = kwargs.get("custom_keys", {})
        self._init_split_parameters()

    def get_optimizer(self):
        raise NotImplementedError

    def _init_split_parameters(self):
        for model in self.models:
            for param_name, parameter in model.named_parameters():
                for key in self.custom_keys.keys():
                    key_in_params_flag = key.strip("~") in param_name
                    forbid_key_in_params_flag = len(key) and key[0] == "~" and (not key.strip("~") in param_name)
                    if key_in_params_flag or forbid_key_in_params_flag:
                        lr_factor = self.custom_keys[key].get("lr_factor", 1)
                        wd_factor = self.custom_keys[key].get("wd_factor", 1)
                        if lr_factor == 0:
                            get_logger().info(f"{param_name} is freezed")
                        self.params.append(
                            {
                                "params": [parameter],
                                "lr": self.lr * lr_factor,
                                "weight_decay": self.wd * wd_factor,
                            }
                        )
                        break
                else:
                    if not self.wd_ignore_bias or "weight" in param_name:
                        self.decay_params.append(parameter)
                    else:
                        self.no_decay_params.append(parameter)
        self.params.append({"params": self.decay_params, "weight_decay": self.wd})
        self.params.append({"params": self.no_decay_params, "weight_decay": 0.0})


@OPTIMIZERS.register_module()
class SGD(BaseOptimizer):
    def __init__(
        self,
        models,
        learning_rate=0.001,
        momentum=0.9,
        wd=0.0,
        wd_ignore_bias=False,
        **kwargs,
    ):
        super(SGD, self).__init__(models, learning_rate, wd, wd_ignore_bias, **kwargs)
        self.momentum = momentum

    def get_optimizer(self):
        return torch.optim.SGD(
            self.params,
            lr=self.lr,
            momentum=self.momentum,
        )


@OPTIMIZERS.register_module()
class Adam(BaseOptimizer):
    def __init__(
        self,
        models,
        learning_rate=0.001,
        wd=0.0,
        wd_ignore_bias=False,
        **kwargs,
    ):
        super(Adam, self).__init__(models, learning_rate, wd, wd_ignore_bias, **kwargs)

    def get_optimizer(self):
        return torch.optim.Adam(
            self.params,
            lr=self.lr,
        )


@OPTIMIZERS.register_module()
class AdamW(BaseOptimizer):
    def __init__(
        self,
        models,
        learning_rate=0.001,
        betas=(0.9, 0.999),
        wd=0.0,
        wd_ignore_bias=False,
        amsgrad=False,
        **kwargs,
    ):
        super(AdamW, self).__init__(models, learning_rate, wd, wd_ignore_bias, **kwargs)
        self.betas = betas
        self.amsgrad = amsgrad

    def get_optimizer(self):
        return torch.optim.AdamW(
            self.params,
            lr=self.lr,
            betas=self.betas,
            amsgrad=self.amsgrad,
            eps=1e-3,
        )
