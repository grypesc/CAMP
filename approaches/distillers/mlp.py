import torch
import torch.nn.functional as F

from torch import nn


from .base import BaseDistiller

class MLPDistiller(BaseDistiller):
    def __init__(self, distill_loss_fn: str, num_features: int, **kwargs):
        super().__init__(distill_loss_fn)
        distill_proj_hidden_dim = num_features
        self.distill_projector = nn.Sequential(
            nn.Linear(num_features, distill_proj_hidden_dim),
            nn.BatchNorm1d(distill_proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(distill_proj_hidden_dim, num_features),
        )

    def calculate_single_view_loss(self, new_feats: torch.Tensor, old_feats: torch.Tensor):
        projected_new_feats = self.distill_projector(new_feats)
        return self.loss_fn(projected_new_feats, old_feats)
