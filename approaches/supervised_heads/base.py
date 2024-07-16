import abc
import torch


class BaseSupervisedHead(abc.ABC, torch.nn.Module):
    requires_multiview_transform = False
    returns_logits = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, features: torch.Tensor, targets: torch.Tensor):
        pass
