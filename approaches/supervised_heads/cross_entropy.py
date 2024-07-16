import torch
import torch.nn.functional as F

from .base import BaseSupervisedHead

class CrossEntropyHead(BaseSupervisedHead):
    returns_logits = True

    def __init__(self, num_features, num_classes, **kwargs) -> None:
        super().__init__()
        self.head = torch.nn.Linear(num_features, num_classes)

    def forward(self, features: torch.Tensor, targets: torch.Tensor):
        logits = self.head(features)
        loss = F.cross_entropy(logits, targets)
        return loss, logits
