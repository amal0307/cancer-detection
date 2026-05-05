import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        s = self.last_epoch
        if s < self.warmup_steps:
            scale = s / max(self.warmup_steps, 1)
        else:
            p = (s - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * p))
        return [base_lr * scale for base_lr in self.base_lrs]