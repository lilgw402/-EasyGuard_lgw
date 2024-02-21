# coding=utf-8
# Email: jiangxubin@bytedance.com
# Create: 2023/3/9 17:58 下午
import inspect


class Registry:
    """A registry to map strings to classes"""

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + f"(name={self._name}, " f"items={self._module_dict})"
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        """Get the registry record"""
        if key not in self._module_dict:
            raise KeyError("{} is not registered!".format(key))
        return self._module_dict[key]

    def register_module(self, name=None, override=True, module=None):
        """Register a module"""
        if not isinstance(override, bool):
            raise TypeError(f"force must be a boolean, but got {type(override)}")

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module_class=module, module_name=name, override=override)
            return module

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(f"name must be a str, but got {type(name)}")

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(module_class=cls, module_name=name, override=override)
            return cls

        return _register

    def _register_module(self, module_class, module_name=None, override=True):
        if not inspect.isclass(module_class):
            raise TypeError("module must be a class, " f"but got {type(module_class)}")

        if module_name is None:
            module_name = module_class.__name__

        if not override and module_name in self._module_dict:
            raise KeyError(f"{module_name} is already registered " f"in {self.name}")
        self._module_dict[module_name] = module_class


MODELS = Registry("model")
DATASETS = Registry("dataset")
FEATURE_PROVIDERS = Registry("feature_provider")
METRICS = Registry("metric")
SCHEDULERS = Registry("lr_scheduler")
OPTIMIZERS = Registry("optimizer")

DEFAULT_OPTIMIZER = "AdamW"
DEFAULT_SCHEDULER = "LinearLrSchedulerWithWarmUp"


def get_model_module(model_type):
    model = MODELS.get(model_type)
    return model


def get_data_module(data_module):
    dataset = DATASETS.get(data_module)
    return dataset


def get_metric_instance(metric_type, arg_dict):
    metric = METRICS.get(metric_type)(**arg_dict)
    return metric


def build_lr_scheduler_instance(optimizer, arg_dict):
    lr_scheduler_type = arg_dict.pop("scheduler_name", DEFAULT_SCHEDULER)
    lr_scheduler = SCHEDULERS.get(lr_scheduler_type)(optimizer, **arg_dict)
    return lr_scheduler


def build_optimizer_instance(model_list, arg_dict):
    optimizer_type = arg_dict.get("optimizer_name", DEFAULT_OPTIMIZER)
    optimizer = OPTIMIZERS.get(optimizer_type)(model_list, **arg_dict).get_optimizer()
    return optimizer
