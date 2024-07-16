import torch
import torch.nn.functional as F

from .base import BaseDistiller


class FeatureDistiller(BaseDistiller):
    def __init__(self, distill_loss_fn: str, **kwargs):
        super().__init__(distill_loss_fn)

    def calculate_single_view_loss(self, new_feats: torch.Tensor, old_feats: torch.Tensor):
        return self.loss_fn(new_feats, old_feats)
