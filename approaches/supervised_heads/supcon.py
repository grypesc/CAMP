from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from .base import BaseSupervisedHead
from utils.losses import SupConLoss


class SupConHead(BaseSupervisedHead):
    requires_multiview_transform = True

    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 **kwargs
    ) -> None:
        super().__init__()

        head_hidden_dim = 2048
        head_output_dim = 128

        self.head = nn.Sequential(
            nn.Linear(num_features, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, head_output_dim),
        )

        self.loss_fn = SupConLoss()

    def forward(self, features: List[torch.Tensor], targets: torch.Tensor):
        out = [self.head(feat) for feat in features]
        out = [F.normalize(feat, dim=-1) for feat in out]
        out = torch.stack(out, dim=1)
        loss = self.loss_fn(out, labels=targets)
        return loss, out
