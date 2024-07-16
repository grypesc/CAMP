import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ExponentialLR


class WarmUpExponentialLR(nn.Module):
    """Warm-up and exponential decay chain scheduler. If warmup_iters > 0 than warm-ups linearly for warmup_iters iterations.
    Then it decays the learning rate every epoch. It is a good idea to set warmup_iters as total number of samples in epoch / batch size.
    Good for transformers"""

    def __init__(self, optimizer, lr_decay=0.97, warmup_iters=0):
        super().__init__()
        self.total_steps, self.warmup_iters = 0, warmup_iters
        self.warmup_scheduler = LinearLR(optimizer, 1e-8, total_iters=warmup_iters) if warmup_iters else None
        self.decay_scheduler = ExponentialLR(optimizer, gamma=lr_decay, last_epoch=-1)

    def step_iter(self):
        self.total_steps += 1
        if self.warmup_scheduler:
            self.warmup_scheduler.step()

    def step_epoch(self):
        if self.total_steps > self.warmup_iters:
            self.decay_scheduler.step()


class WarmUpCosineAnnealingLR(nn.Module):
    """Warm-up and exponential decay chain scheduler. If warmup_iters > 0 than warm-ups linearly for warmup_iters iterations.
    Then it decays the learning rate every epoch. It is a good idea to set warmup_iters as total number of samples in epoch / batch size.
    Good for transformers"""

    def __init__(self, optimizer, epochs, eta_min, warmup_iters=0):
        super().__init__()
        self.total_steps, self.warmup_iters = 0, warmup_iters
        self.warmup_scheduler = LinearLR(optimizer, 1e-8, total_iters=warmup_iters) if warmup_iters else None
        self.decay_scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=eta_min)

    def step_iter(self):
        self.total_steps += 1
        if self.warmup_scheduler:
            self.warmup_scheduler.step()

    def step_epoch(self):
        if self.total_steps > self.warmup_iters:
            self.decay_scheduler.step()
