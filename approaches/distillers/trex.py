import torch
import torch.nn.functional as F

from torch import nn

from .base import BaseDistiller


class TrexDistiller(BaseDistiller):
    def __init__(self, distill_loss_fn: str, num_features: int, **kwargs):
        super().__init__(distill_loss_fn)
        distill_proj_hidden_dim = num_features
        self.distill_projector = nn.Sequential(
            nn.Linear(num_features, 2*distill_proj_hidden_dim),
            nn.BatchNorm1d(2*distill_proj_hidden_dim),
            nn.GELU(),
            nn.Linear(2*distill_proj_hidden_dim, num_features)
        )

    def calculate_single_view_loss(self, new_feats: torch.Tensor, old_feats: torch.Tensor):
        new_feats = F.normalize(new_feats, dim=1)
        projected_new_feats = self.distill_projector(new_feats)
        projected_new_feats = F.normalize(projected_new_feats, dim=1)
        return self.loss_fn(projected_new_feats, old_feats)
