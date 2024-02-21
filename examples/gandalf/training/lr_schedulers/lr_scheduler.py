# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-03-17 14:42:41
# Modified: 2023-03-17 14:42:41
import math

import torch.optim
from torch.optim.lr_scheduler import _LRScheduler
from utils.registry import SCHEDULERS


class WarmUpLrScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        last_epoch,
        verbose,
        warmup_steps,
    ):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.warmup_steps = warmup_steps
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        for group in optimizer.param_groups:
            group.setdefault("init_lr", group["lr"])
        self._step_count = 0

    def get_lr(self):
        return []

    def step(self, epoch=None):
        self._step_count += 1
        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group["lr"] = lr


@SCHEDULERS.register_module()
class ConstantLrSchedulerWithWarmUp(WarmUpLrScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1, verbose=False, **kwargs):
        super(ConstantLrSchedulerWithWarmUp, self).__init__(optimizer, last_epoch, verbose, warmup_steps)

    def get_lr(self):
        if 0 < self._step_count < self.warmup_steps:
            return [base_lr * self._step_count / self.warmup_steps for base_lr in self.base_lrs]
        else:
            return self.base_lrs


@SCHEDULERS.register_module()
class LinearLrSchedulerWithWarmUp(WarmUpLrScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps=1000,
        last_epoch=-1,
        verbose=False,
        start_factor=1.0,
        end_factor=1.0 / 3,
        total_iters=5,
        **kwargs,
    ):
        super(LinearLrSchedulerWithWarmUp, self).__init__(optimizer, last_epoch, verbose, warmup_steps)
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        self.warmup_epoch = 0

    def get_lr(self):
        if 0 <= self._step_count < self.warmup_steps:
            return [base_lr * self._step_count / self.warmup_steps for base_lr in self.base_lrs]
        if self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            base_lr
            * (
                1
                + (self.end_factor - self.start_factor)
                * (self.last_epoch - self.warmup_epoch)
                / (self.total_iters - self.warmup_epoch)
            )
            for base_lr in self.base_lrs
        ]


@SCHEDULERS.register_module()
class CosineLrSchedulerWithWarmUp(WarmUpLrScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
        verbose=False,
        num_cycles: float = 0.5,
        min_factor: float = 0.0,
        total_iters=5,
        **kwargs,
    ):
        super(CosineLrSchedulerWithWarmUp, self).__init__(optimizer, last_epoch, verbose, warmup_steps)
        self.num_cycles = num_cycles
        self.min_factor = min_factor
        self.total_iters = total_iters

    def get_lr(self):
        if 0 <= self._step_count < self.warmup_steps:
            return [base_lr * self._step_count / self.warmup_steps for base_lr in self.base_lrs]
        if self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]
        progress = float(self._step_count - self.warmup_steps) / float(max(1, self.total_iters - self.warmup_steps))
        return max(
            0.0,
            self.min_factor
            + 0.5 * (1.0 - self.min_factor) * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)),
        )


@SCHEDULERS.register_module()
class ExponentialLrSchedulerWithWarmUp(WarmUpLrScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        last_epoch=-1,
        verbose=False,
        gamma=1.0,
        total_iters=5,
        **kwargs,
    ):
        super(ExponentialLrSchedulerWithWarmUp, self).__init__(optimizer, last_epoch, verbose, warmup_steps)
        self.gamma = gamma
        self.total_iters = total_iters

    def get_lr(self):
        if 0 <= self._step_count < self.warmup_steps:
            return [base_lr * self._step_count / self.warmup_steps for base_lr in self.base_lrs]
        if self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]
