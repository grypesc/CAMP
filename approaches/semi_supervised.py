import logging
import torch
import numpy as np

from copy import deepcopy
from tqdm import tqdm
from torchmetrics import MeanMetric

from .base import BaseApproach
from datasets.base import DataLoaderCyclicIterator


class SemiSupervised(BaseApproach):

    def __init__(self, args, datamodule, device):
        super().__init__(args, datamodule, device)
        self.n_views = 2

    def train_feature_extractor(self, t, train_dl_dict, valid_dl_dict):
        device = self.device

        labeled_dl = deepcopy(train_dl_dict['known_class_labeled'])
        labeled_dl = self.add_dataset_to_loader(labeled_dl, self.exemplar_dataset)
        if self.supervised_head_cls.requires_multiview_transform:
            labeled_dl.dataset.transforms = train_dl_dict['ssl'].dataset.transforms

        num_known_classes = np.unique(train_dl_dict["known_class"].dataset.targets).shape[0]
        num_novel_classes = np.unique(train_dl_dict["novel_class"].dataset.targets).shape[0]

        self_supervised_head = self.self_supervised_head_cls(self.num_features, **self.args.__dict__).to(device)
        supervised_head = self.supervised_head_cls(self.num_features, self.total_classes, num_known_classes=num_known_classes, num_novel_classes=num_novel_classes, t=t, **self.args.__dict__).to(device)
        self.old_backbone = deepcopy(self.backbone)

        params = list(self.backbone.parameters()) + \
                 list(self_supervised_head.parameters()) + \
                 list(supervised_head.parameters())
        if t > 0:
            distiller = self.distill_cls(num_features=self.num_features, **self.args.__dict__).to(device)
            params += list(distiller.parameters())

        logging.info(f'The model has {sum(p.numel() for p in params if p.requires_grad):,} trainable parameters')
        logging.info(f'The model has {sum(p.numel() for p in params if not p.requires_grad):,} frozen parameters')
        optimizer, lr_scheduler = self.get_optimizer(params)

        for epoch in tqdm(range(self.epochs)):
            self.backbone.train()
            supervised_head.train()
            metrics = {
                "train_total_loss": MeanMetric(),
                "train_ssl_loss": MeanMetric(),
                "train_sl_loss": MeanMetric(),
                "train_distill_loss": MeanMetric(),
            }

            labeled_dl_iter = DataLoaderCyclicIterator(labeled_dl)

            for X_ssl in train_dl_dict['ssl']:
                # SSL
                features_ssl = self.nview_forward(self.backbone, X_ssl)
                self_supervised_loss = self_supervised_head.calculate_loss(features_ssl)
                # SL
                X_labeled, Y_labeled = next(labeled_dl_iter)
                Y_labeled = Y_labeled.to(self.device, non_blocking=True)
                features_labeled = self.nview_forward(self.backbone, X_labeled)
                supervised_loss, _ = supervised_head(features_labeled, Y_labeled)

                loss = (1 - self.beta) * self_supervised_loss + self.beta * supervised_loss

                # Distill
                if t > 0:
                    with torch.no_grad():
                        old_features_ssl = self.nview_forward(self.old_backbone, X_ssl)
                    distill_loss = distiller(features_ssl, old_features_ssl)
                    loss = (1 - self.alpha) * loss + self.alpha * distill_loss
                    metrics["train_distill_loss"].update(distill_loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.clip)
                optimizer.step()

                metrics["train_total_loss"].update(loss.item())
                metrics["train_ssl_loss"].update(self_supervised_loss.item())
                metrics["train_sl_loss"].update(supervised_loss.item())
                lr_scheduler.step_iter()
            lr_scheduler.step_epoch()

            s = f"Epoch: {epoch} "
            for k, v in metrics.items():
                s += f"{k}: {v.compute():.2f} "
            logging.info(s)
