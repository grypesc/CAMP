import torch

from typing import List


class BaseSelfSupervisedHead(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def calculate_loss(self, features: List[torch.Tensor]) -> torch.Tensor:
        return torch.tensor([0.0], device=features[0].device)
