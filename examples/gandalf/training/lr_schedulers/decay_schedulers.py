# -*- coding:utf-8 -*-
#
# Copyright (c) 2020 Bytedance.com, Inc. All Rights Reserved
#
# Each engineer has a duty to keep the code elegant
#
"""
FileName: decay_schedulers.py
Author: Ye Jinxing (yejinxing.yjx@bytedance.com)
Created Time: 2021-06-19 15:10:12
"""
import functools

import numpy as np
import torch
from utils.registry import SCHEDULERS


def raw_with_warm_up(learning_rate, warmup_steps=0):
    def decorator(func):
        r"""
        func: python function
            args:
                - step
                    current step count.
                - total_step
                    step amount during this training.
            kw args:
                whatever, customized args.
        """

        def warm_up(*args, **kw):
            step = args[0]
            total_step = kw.get("_total_step", -1)
            if warmup_steps > 0 and step <= warmup_steps:
                return learning_rate * np.min(((step / warmup_steps), 1.0))
            else:
                total_step = np.max((0, total_step - warmup_steps))
                return func(step - warmup_steps, np.max((total_step, warmup_steps + 1)))

        return warm_up

    return decorator


class BaseLrScheduler(object):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_step: int,
        warmup_steps: int,
        **kwargs,
    ):
        r"""
        params:
            - optimizer: torch.optim.Optimizer
                the optimizer whose learning rate is managed by this scheduler.
            - total_step: int
                training step amount, set as -1 if unavailable.
            - warm_up_steps: int
                warm up step configure, make lr increase from 0 to default lr in $warm_up_steps$ steps.
                set as 0 to forbid warm-up.
        """
        self._optimizer = optimizer
        self._total_step = total_step
        self._base_lr = np.max([x.get("lr", 0.0) for x in self._optimizer.param_groups])
        self._latest_lr = self._base_lr
        self._lr_decay_func = lambda _: self._latest_lr
        self._warmup_steps = warmup_steps
        self.with_warm_up = functools.partial(
            raw_with_warm_up,
            learning_rate=self._base_lr,
            warmup_steps=warmup_steps,
        )()

    @property
    def optimizer(self):
        return self._optimizer

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if key in ("_total_step", "_base_lr", "_latest_lr", "_warmup_steps")
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        return self._latest_lr

    def update_lr(self, step):
        new_lr = self._lr_decay_func(step)
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = new_lr
        self._latest_lr = new_lr

    def get_and_update_lr(self, step):
        self.update_lr(step)
        return self._latest_lr


@SCHEDULERS.register_module()
class ConstLrScheduler(BaseLrScheduler):
    def __init__(self, optimizer, total_step=-1, warmup_steps=0, **kwargs):
        r"""
        Keep lr const during the training. No other custom args.
        """
        super(ConstLrScheduler, self).__init__(optimizer, total_step, warmup_steps, **kwargs)

        @self.with_warm_up
        def f(_step, _total_step=-1):
            return self._base_lr

        self._lr_decay_func = functools.partial(f, _total_step=total_step)


@SCHEDULERS.register_module()
class LinearDecayLrScheduler(BaseLrScheduler):
    def __init__(self, optimizer, total_step=-1, warmup_steps=0, min_lr=0.0, **kwargs):
        r"""
        Make lr linear decay to target $min_lr$ along each step.
        params:
            - min_lr
                learning rate at the last step.
        """
        super(LinearDecayLrScheduler, self).__init__(optimizer, total_step, warmup_steps, **kwargs)

        @self.with_warm_up
        def f(_step, _total_step=-1):
            r"""
            When you try to refactor those codes, DO NOT remove $_total_step$ param.
            This parameter will by modified at warm-up stage.
            """
            return np.max(
                (
                    min_lr,
                    (1 - _step / (_total_step + 1e-7)) * self._base_lr,
                )
            )

        self._lr_decay_func = functools.partial(f, _total_step=total_step)


@SCHEDULERS.register_module()
class CosineDecayLrScheduler(BaseLrScheduler):
    def __init__(self, optimizer, total_step=-1, warmup_steps=0, min_lr=0.0, **kwargs):
        r"""
        Make lr cosine decay to 0.0, until it reaches target $min_lr$ along each step.
        params:
            - min_lr
                learning rate at the last step.
        """
        super(CosineDecayLrScheduler, self).__init__(optimizer, total_step, warmup_steps, **kwargs)

        @self.with_warm_up
        def f(_step, _total_step=-1):
            r"""
            When you try to refactor those codes, DO NOT remove $_total_step$ param.
            This parameter will be modified at warm-up stage.
            """
            return np.max(
                (
                    min_lr,
                    self._base_lr * np.cos(0.5 * _step * np.pi / (_total_step + 1e-7)),
                )
            )

        self._lr_decay_func = functools.partial(f, _total_step=total_step)


@SCHEDULERS.register_module()
class ExpDecayLrScheduler(BaseLrScheduler):
    def __init__(
        self,
        optimizer,
        total_step=-1,
        warmup_steps=0,
        min_lr=0.0,
        gamma=1.0,
        decay_step=1,
        **kwargs,
    ):
        r"""
        Make lr times $gamma$ for each $decay_step$ steps, until it becomes $min_lr$.
        params:
            - min_lr
            - gamma
            - decay_step
        """
        super(ExpDecayLrScheduler, self).__init__(optimizer, total_step, warmup_steps, **kwargs)

        @self.with_warm_up
        def f(_step, _total_step=-1):
            r"""
            When you try to refactor those codes, DO NOT remove $_total_step$ param.
            This parameter will by modified at warm-up stage.
            """
            return np.max(
                (
                    min_lr,
                    self._base_lr * (gamma ** (_step // decay_step)),
                )
            )

        self._lr_decay_func = functools.partial(f, _total_step=total_step)
