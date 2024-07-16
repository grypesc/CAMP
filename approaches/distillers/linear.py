import torch
import torch.nn.functional as F

from torch import nn

from .base import BaseDistiller


class LinearDistiller(BaseDistiller):
    def __init__(self, distill_loss_fn: str, num_features: int, **kwargs):
        super().__init__(distill_loss_fn)
        self.distill_projector = nn.Linear(num_features, num_features)

    def calculate_single_view_loss(self, new_feats: torch.Tensor, old_feats: torch.Tensor):
        projected_new_feats = self.distill_projector(new_feats)
        return self.loss_fn(projected_new_feats, old_feats)
