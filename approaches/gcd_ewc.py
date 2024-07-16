import itertools
import logging
import torch

from copy import deepcopy
from tqdm import tqdm
from torchmetrics import MeanMetric

from .semi_supervised import SemiSupervised
from datasets.base import DataLoaderCyclicIterator


class GCD_EWC(SemiSupervised):
    """ Taken from FACIL"""

    def __init__(self, args, datamodule, device):
        super().__init__(args, datamodule, device)
        self.older_params = {n: p.clone().detach() for n, p in self.backbone.named_parameters() if p.requires_grad}
        # Store fisher information weight importance
        self.fisher = {n: torch.zeros(p.shape, device=self.device) for n, p in self.backbone.named_parameters()
                       if p.requires_grad}
        self.lamb = args.lamb

    @classmethod
    def add_argparse_args(cls, parser):
        SemiSupervised.add_argparse_args(parser)
        parser.add_argument('--lamb',
                            help='learning-rate',
                            type=float,
                            default=5000)
        return parser

    def train_feature_extractor(self, t, train_dl_dict, valid_dl_dict):
        device = self.device

        self_supervised_head = self.self_supervised_head_cls(self.num_features, **self.args.__dict__).to(device)
        supervised_head = self.supervised_head_cls(self.num_features, num_classes=self.total_classes, **self.args.__dict__).to(device)

        supervised_head.train()
        self.old_backbone = deepcopy(self.backbone)

        params = list(self.backbone.parameters()) + \
                 list(self_supervised_head.parameters()) + \
                 list(supervised_head.parameters())

        if t > 0:
            distiller = self.distill_cls(num_features=self.num_features, **self.args.__dict__).to(device)
            distiller.train()
            params += list(distiller.parameters())

        logging.info(f'The model has {sum(p.numel() for p in params if p.requires_grad):,} trainable parameters')
        logging.info(f'The model has {sum(p.numel() for p in params if not p.requires_grad):,} frozen parameters')
        optimizer, lr_scheduler = self.get_optimizer(params)

        labeled_dl = deepcopy(train_dl_dict['known_class_labeled'])
        labeled_dl = self.add_dataset_to_loader(labeled_dl, self.exemplar_dataset)
        if self.supervised_head_cls.requires_multiview_transform:
            labeled_dl.dataset.transforms = train_dl_dict['ssl'].dataset.transforms

        for epoch in tqdm(range(self.epochs)):
            self.backbone.train()
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
                supervised_loss, supervised_out = supervised_head(features_labeled, Y_labeled)

                loss = (1 - self.beta) * self_supervised_loss + self.beta * supervised_loss
                if t > 0:
                    # EWC distill loss
                    distill_loss = self.distill_loss(t)
                    # Additional distill loss
                    with torch.no_grad():
                        old_features_ssl = self.nview_forward(self.old_backbone, X_ssl)
                    distill_loss += distiller(features_ssl, old_features_ssl)
                    loss = (1 - self.alpha) * loss + self.alpha * distill_loss
                    loss += distill_loss
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

        self.backbone.fc = supervised_head.head
        optimizer.zero_grad()
        self.post_train(t, train_dl_dict)
        self.backbone.fc = torch.nn.Identity()

    def compute_fisher_matrix_diag(self, trn_loader):
        # Store Fisher Information
        trn_loader = trn_loader["known_class_labeled"]
        fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self.backbone.named_parameters()
                  if p.requires_grad}
        # Compute fisher information for specified number of samples -- rounded to the batch size
        num_samples = len(trn_loader.dataset)
        n_samples_batches = (num_samples // trn_loader.batch_size + 1) if num_samples > 0 \
            else (len(trn_loader.dataset) // trn_loader.batch_size)
        # Do forward and backward pass to compute the fisher information
        self.backbone.train()
        for images, targets in itertools.islice(trn_loader, n_samples_batches):
            outputs = self.backbone(images.to(self.device))
            preds = outputs.argmax(1)
            loss = torch.nn.functional.cross_entropy(outputs, preds)
            # self.optimizer.zero_grad()
            loss.backward()
            # Accumulate all gradients from loss with regularization
            for n, p in self.backbone.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) * len(targets)
        # Apply mean across all samples
        n_samples = n_samples_batches * trn_loader.batch_size
        fisher = {n: (p / n_samples) for n, p in fisher.items()}
        return fisher

    def post_train(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Store current parameters for the next task
        self.older_params = {n: p.clone().detach() for n, p in self.backbone.named_parameters() if p.requires_grad}

        # calculate Fisher information
        curr_fisher = self.compute_fisher_matrix_diag(trn_loader)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self.fisher.keys():
            # Added option to accumulate fisher over time with a pre-fixed growing alpha
            if self.alpha == -1:
                alpha = (sum(self.backbone.task_cls[:t]) / sum(self.backbone.task_cls)).to(self.device)
                self.fisher[n] = alpha * self.fisher[n] + (1 - alpha) * curr_fisher[n]
            else:
                self.fisher[n] = (self.alpha * self.fisher[n] + (1 - self.alpha) * curr_fisher[n])

    def distill_loss(self, t):
        """Returns the distillation loss value"""
        distil_loss = 0
        if t > 0:
            loss_reg = torch.zeros((1,), device=self.device)
            # Eq. 3: elastic weight consolidation quadratic penalty
            for n, p in self.backbone.named_parameters():
                if n in self.fisher.keys():
                    loss_reg += torch.sum(self.fisher[n] * (p - self.older_params[n]).pow(2)) / 2
            distil_loss += self.lamb * loss_reg
        return distil_loss
