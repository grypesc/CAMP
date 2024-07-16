import torch

import torch.nn as nn
import torch.nn.functional as F


def binarize(T, num_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, num_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class ProxyAnchorHead(torch.nn.Module):
    requires_multiview_transform = False

    def __init__(self, sz_embed, proxies, num_known_classes, num_novel_classes, t, mrg=0.1, num_exemplars=0, **kwargs):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        if num_exemplars <= 0:
            raise RuntimeError("PA requires exemplars to run")
        self.proxies = proxies
        self.num_known_classes = num_known_classes
        self.num_novel_classes = num_novel_classes
        self.num_all_labeled = (t+1) * num_known_classes
        self.head = nn.Linear(sz_embed, self.num_all_labeled)

        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = 32

    def forward(self, X, T):
        # Calculate offset caused by novel classes
        task = torch.floor(T / (self.num_known_classes + self.num_novel_classes)).long()
        T -= task * self.num_novel_classes

        # logits = self.head(X)
        # ce_loss = nn.functional.cross_entropy(logits, T)

        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=T, num_classes=self.num_all_labeled)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.num_all_labeled
        loss = pos_term + neg_term #+ ce_loss

        return loss, None
