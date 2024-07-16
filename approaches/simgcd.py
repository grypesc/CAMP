import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from copy import deepcopy
from tqdm import tqdm
from torchmetrics import MeanMetric

from datasets.base import DataLoaderCyclicIterator, get_all_tasks_dataloader
from utils.cluster_and_log_utils import log_accs_from_preds
from utils.losses import SupConLoss, info_nce_logits
from .base import BaseApproach


class SimGCD(BaseApproach):
    """ Taken from original paper https://arxiv.org/pdf/2304.14310.pdf"""

    def __init__(self, args, datamodule, device):
        super().__init__(args, datamodule, device)
        self.n_views = 2
        self.warmup_teacher_temp_epochs = args.warmup_teacher_temp_epochs
        self.epochs = args.epochs
        self.warmup_teacher_temp = args.warmup_teacher_temp
        self.teacher_temp = args.teacher_temp
        self.memax_weight = args.memax_weight
        self.student = nn.Sequential(self.backbone, DINOIGCDHead(384, self.total_classes).to(self.device))
        self.num_novel_classes = datamodule.novel_classes_per_task
        self.num_all_classes = datamodule.classes_per_task


    @classmethod
    def add_argparse_args(cls, parser):
        BaseApproach.add_argparse_args(parser)
        parser.add_argument('--warmup_teacher_temp_epochs',
                            type=float,
                            default=3)
        parser.add_argument('--warmup_teacher_temp',
                            type=float,
                            default=0.07)
        parser.add_argument('--teacher_temp',
                            type=float,
                            default=0.04)
        parser.add_argument('--memax_weight',
                            type=float,
                            default=2)
        return parser

    def train_feature_extractor(self, t, train_dl_dict, valid_dl_dict):
        cluster_criterion = DistillLoss(
            self.warmup_teacher_temp_epochs,
            self.epochs,
            self.n_views,
            self.warmup_teacher_temp,
            self.teacher_temp,
        )

        labeled_dl = deepcopy(train_dl_dict['known_class_labeled'])
        labeled_dl = self.add_dataset_to_loader(labeled_dl, self.exemplar_dataset)
        labeled_dl.dataset.transforms = train_dl_dict['ssl'].dataset.transforms
        self.old_backbone = deepcopy(self.backbone)

        params = list(self.student.parameters()) + list(cluster_criterion.parameters())
        if t > 0:
            distiller = self.distill_cls(num_features=self.num_features, **self.args.__dict__).to(self.device)
            params += list(distiller.parameters())
            distiller.train()
        optimizer, lr_scheduler = self.get_optimizer(params)
        for epoch in tqdm(range(self.epochs)):
            self.student.train()
            metrics = {
                "train_total_loss": MeanMetric(),
                "train_ssl_loss": MeanMetric(),
                "train_sl_loss": MeanMetric(),
                "train_distill_loss": MeanMetric(),
            }

            labeled_dl_iter = DataLoaderCyclicIterator(labeled_dl)
            for X_ssl in train_dl_dict["ssl"]:
                # SSL
                ssl_bsz = X_ssl[0].shape[0]
                ssl_labels = torch.full((ssl_bsz,), fill_value=-1, dtype=torch.int, device=self.device)

                # SL
                X_labeled, Y_labeled = next(labeled_dl_iter)
                sl_bsz = X_labeled[0].shape[0]
                Y_labeled = Y_labeled.to(self.device, non_blocking=True)

                images = torch.cat((X_ssl[0], X_labeled[0], X_ssl[1], X_labeled[1]), dim=0).to(self.device)
                class_labels = torch.cat((ssl_labels, Y_labeled), dim=0).to(self.device)
                mask_lab = class_labels != -1

                student_proj, student_out, features = self.student(images)
                teacher_out = student_out.detach()

                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # clustering, unsup
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += self.memax_weight * me_max_loss

                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(student_proj, student_proj.shape[0] // 2)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                loss_sl = cls_loss + sup_con_loss
                loss_ssl = contrastive_loss + cluster_loss
                loss = self.beta * loss_sl + (1 - self.beta) * loss_ssl

                #Distillation loss
                if t >= 1:
                    with torch.no_grad():
                        old_features_ssl = self.nview_forward(self.old_backbone, images)
                    distill_loss = distiller(features, old_features_ssl)
                    loss = (1 - self.alpha) * loss + self.alpha * distill_loss
                    metrics["train_distill_loss"].update(distill_loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.clip)
                optimizer.step()

                metrics["train_total_loss"].update(loss.item())
                metrics["train_ssl_loss"].update(loss_ssl.item())
                metrics["train_sl_loss"].update(loss_sl.item())

                lr_scheduler.step_iter()
            lr_scheduler.step_epoch()

            s = f"Epoch: {epoch} "
            for k, v in metrics.items():
                s += f"{k}: {v.compute():.2f} "
            logging.info(s)


    def eval_task(self, train_dl_dict, valid_dl_dict, t, old_prototypes, adapted_prototypes):
        """ Eval the approach task agnostic and task aware"""
        ### Classification accuracy ###

        results = {"tag_acc": {}, "taw_acc": {}, "tag_known_acc": {}, "taw_known_acc": {}, "tag_novel_acc": {}, "taw_novel_acc": {}}
        for i in range(t + 1):
            all_acc, old_acc, new_acc = self.test(self.student, valid_dl_dict[i]["known_class"], valid_dl_dict[i]["novel_class"])

            results["tag_acc"][i], results["taw_acc"][i] = all_acc, -1
            results["tag_known_acc"][i], results["taw_known_acc"][i] = old_acc, -1
            results["tag_novel_acc"][i], results["taw_novel_acc"][i] = new_acc, -1

            for key in ["tag_acc", "taw_acc", "tag_known_acc", "taw_known_acc", "tag_novel_acc", "taw_novel_acc"]:
                results[f"avg_{key}"] = np.mean([acc for acc in results[key].values()])

        # NMC on all_foreground as an upper bound for acc
            all_foreground_train_dl = get_all_tasks_dataloader(train_dl_dict, 'all_foreground', t)
            all_foreground_valid_dl = get_all_tasks_dataloader(valid_dl_dict, 'all_foreground', t)
            val_transform = valid_dl_dict[0]["known_class_labeled"].dataset.transforms
            all_foreground_prototypes = {0: self.calculate_prototypes(all_foreground_train_dl, val_transform)}

            results["upper_bound_tag_acc"], results["upper_bound_taw_acc"] = {}, {}
            for j in range(t+1):
                results["upper_bound_tag_acc"][j], results["upper_bound_taw_acc"][j] = self.predict_acc_nmc(
                    all_foreground_prototypes, valid_dl_dict[j]["all_foreground"], j)
                results["avg_upper_bound_tag_acc"] = np.mean([acc for acc in results["upper_bound_tag_acc"].values()])
                results["avg_upper_bound_taw_acc"] = np.mean([acc for acc in results["upper_bound_taw_acc"].values()])
        return results

    def test(self, model, test_known_loader, test_novel_loader):
        model.eval()

        preds, targets = [], []
        mask = np.array([])
        for images, label in test_known_loader:
            images = images.cuda(non_blocking=True)
            with torch.no_grad():
                _, logits, _ = model(images)
                preds.append(logits.argmax(1).cpu().numpy())
                targets.append(label.cpu().numpy())
                mask = np.append(mask, np.full((logits.shape[0]), fill_value=True))
        for images, label in test_novel_loader:
            images = images.cuda(non_blocking=True)
            with torch.no_grad():
                _, logits, _ = model(images)
                preds.append(logits.argmax(1).cpu().numpy())
                targets.append(label.cpu().numpy())
                mask = np.append(mask, np.full((logits.shape[0]), fill_value=False))

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask)

        return all_acc, old_acc, new_acc


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs,
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss


class DINOIGCDHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        ori_x = x
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x)
        return x_proj, logits, ori_x