import torch
from typing import List, Union


class BaseDistiller(torch.nn.Module):
    def __init__(self, distill_loss_fn: str, *args, **kwargs):
        super().__init__()

        if distill_loss_fn == 'mse':
            self.loss_fn = torch.nn.MSELoss()
        elif distill_loss_fn == 'cosine':
            self.loss_fn = self._cosine_similarity_loss
        else:
            raise RuntimeError()

    def forward(self, new_features: Union[torch.Tensor, List[torch.Tensor]],
                old_features: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(new_features, torch.Tensor) and isinstance(old_features, torch.Tensor):
            return self.calculate_single_view_loss(new_features, old_features)
        elif isinstance(new_features, list) and isinstance(old_features, list):
            losses = [self.calculate_single_view_loss(nf, of) for nf, of in zip(new_features, old_features)]
            return torch.stack(losses, dim=0).sum(dim=0)
        else:
            raise NotImplementedError()

    def calculate_single_view_loss(self, new_features: torch.Tensor, old_features: torch.Tensor):
        return torch.tensor([0.0], device=new_features.device)
    
    def _cosine_similarity_loss(self, new_feats: torch.Tensor, old_feats: torch.Tensor):
        return torch.nn.CosineEmbeddingLoss()(new_feats, old_feats, torch.tensor([1], device=new_feats.device))
