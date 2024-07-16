from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from utils.losses import info_nce_logits

from .base import BaseSelfSupervisedHead


class SimCLRHead(BaseSelfSupervisedHead):
    def __init__(self,
                 num_features: int,
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

    def calculate_loss(self, features: List[torch.Tensor]) -> torch.Tensor:
        batch_size_ssl = features[0].shape[0]
        n_views = len(features)

        features = torch.cat(features, dim=0)
        out = self.head(features)
        out = F.normalize(out, dim=-1)
        contrastive_logits, contrastive_labels = info_nce_logits(
            features=out, batch_size=batch_size_ssl, n_views=n_views)
        loss = nn.functional.cross_entropy(contrastive_logits, contrastive_labels)
        return loss
