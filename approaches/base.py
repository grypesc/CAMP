import abc
import copy
import os
import inspect
import random
from tqdm import tqdm
import torch
import logging
import numpy as np
from typing import List, Union
import torch.nn.functional as F

from copy import deepcopy
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.models.resnet import resnet18
from approaches.networks.vit import vit_small
from torchmetrics import MeanMetric, Accuracy
from scipy.optimize import linear_sum_assignment

import approaches.self_supervised_heads
import approaches.supervised_heads
import approaches.distillers

from datasets.base import get_all_tasks_dataloader, MemoryDataset
from utils.kmeans_semi_supervised import SemiSupKMeans
from utils.kmedians_semi_supervised import SemiSupKMedians
from utils.knn import KNN
from utils.kmeans import KMeans, KMedians
from utils.schedulers import WarmUpCosineAnnealingLR, WarmUpExponentialLR
from utils.visualization import visualize_drift


class BaseApproach(abc.ABC, torch.nn.Module):
    self_supervised_heads = {
        k: v for k, v in inspect.getmembers(approaches.self_supervised_heads, inspect.isclass) if
        issubclass(v, torch.nn.Module) and not inspect.isabstract(v)
    }
    supervised_heads = {
        k: v for k, v in inspect.getmembers(approaches.supervised_heads, inspect.isclass) if
        issubclass(v, torch.nn.Module) and not inspect.isabstract(v)
    }
    distillers = {
        k: v for k, v in inspect.getmembers(approaches.distillers, inspect.isclass) if
        issubclass(v, torch.nn.Module) and not inspect.isabstract(v)
    }

    def __init__(self, args, data_module, device):
        super().__init__()

        self.lr = args.lr
        self.lr_scheduler = args.lr_scheduler
        self.lr_decay = args.lr_decay
        self.epochs = args.epochs
        self.weight_decay = args.weight_decay
        self.clip = args.clip
        self.num_exemplars = args.num_exemplars
        self.optimizer = args.optimizer

        self.lp_lr = args.lp_lr
        self.lp_epochs = args.lp_epochs
        self.lp_weight_decay = args.lp_weight_decay
        self.warmup_iters = args.warmup_iters

        self.self_supervised_head_cls = self.self_supervised_heads[args.self_supervised_head]
        self.supervised_head_cls = self.supervised_heads[args.supervised_head]
        self.distill_cls = self.distillers[args.distiller]

        self.beta = args.beta
        self.alpha = args.alpha

        if args.visualize_drift:
            if len(data_module.train_dataset_dict.keys()) != 2:
                logging.warning(f"Visualize drift enabled with {len(data_module.train_dataset_dict.keys())} "
                                "tasks but tested only with 2 tasks")

        self.prototypes = {}
        self.clustering_strategy = args.clustering_strategy
        self.find_centroids = self.find_centroids_semi_sup
        if args.clustering_strategy == "KMeans" or args.clustering_strategy == "KMedians":
            self.find_centroids = self.find_centroids_standard

        if args.network == "vit":
            state_dict = torch.load("pretrained/dino_deitsmall16_pretrain.pth", map_location='cpu')
            self.backbone = vit_small()
            self.backbone.load_state_dict(state_dict)

            if args.visualize_drift:
                self.num_features = 2
                self.backbone.head = torch.nn.Linear(self.backbone.num_features, out_features=2)
            else:
                self.num_features = self.backbone.num_features
                self.backbone.head = torch.nn.Identity()

            for name, param in self.backbone.named_parameters():
                if "blocks.11" not in name:
                    param.requires_grad = False
        else:
            self.backbone = resnet18()
            if data_module.image_size == (32, 32):
                self.backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
                self.backbone.maxpool = torch.nn.Identity()

            if args.visualize_drift:
                self.num_features = 2
                self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, out_features=2)
            else:
                self.num_features = self.backbone.fc.in_features
                self.backbone.fc = torch.nn.Identity()

        self.old_backbone = None
        self.exemplar_dataset = None

        self.total_classes = data_module.total_classes
        self.classes_per_task = data_module.classes_per_task
        self.novel_classes_per_task = data_module.novel_classes_per_task
        self.device = device
        self.args = args

    @classmethod
    def add_argparse_args(cls, parser):
        parser.add_argument('--network',
                            help='Type of backbone network',
                            type=str,
                            default='resnet18',
                            choices=['resnet18', 'vit'])
        parser.add_argument('--lr',
                            help='learning-rate',
                            type=float,
                            default=1e-3)
        parser.add_argument('--lr-scheduler',
                            help='Type of learning rate scheduler',
                            type=str,
                            default='ExponentialLR',
                            choices=['ExponentialLR', 'CosineAnnealingLR'])
        parser.add_argument('--warmup-iters',
                            help='warm up optimizer for this number of iterations',
                            type=int,
                            default=100)
        parser.add_argument('--lr-decay',
                            help='learning-rate',
                            type=float,
                            default=0.95)
        parser.add_argument('--epochs',
                            help='number of epochs',
                            type=int,
                            default=100)
        parser.add_argument('--num-exemplars',
                            help='exemplars per class',
                            type=int,
                            default=0)
        parser.add_argument('--weight-decay',
                            help='weight_decay',
                            type=float,
                            default=0)
        parser.add_argument('--clip',
                            help='gradient clipping',
                            type=float,
                            default=1.0)
        parser.add_argument('--smoothing',
                            help='label smoothing for ce',
                            type=float,
                            default=0.0)
        parser.add_argument('--optimizer',
                            help='optimizer',
                            type=str,
                            default='AdamW',
                            choices=['AdamW', 'SGD'])
        
        # prototypes
        parser.add_argument('--adapt-prototypes',
                            help="Adapt prototypes based on estimated drift",
                            action='store_true')
        parser.add_argument('--adapt-prototypes-epochs',
                            help='number of epochs for training prototypes adapter',
                            type=int,
                            default=100)
        parser.add_argument('--clustering-strategy',
                            help='Algorithm that finds centroids',
                            type=str,
                            default='SemiSupKMeans',
                            choices=['SemiSupKMeans', 'SemiSupKMedians', 'KMeans', 'KMedians'])

        parser.add_argument('--adapt-prototypes-algo',
                            help='Algorithm that finds centroids',
                            type=str,
                            default='ours',
                            choices=['ours', 'SDC'])

        # self-supervised head
        parser.add_argument('--self-supervised-head',
                            help='Type of self-supervised head',
                            type=str,
                            default='BaseSelfSupervisedHead',
                            choices=cls.self_supervised_heads.keys())

        # supervised head
        parser.add_argument('--supervised-head',
                            help='Type of supervised head',
                            type=str,
                            default='CrossEntropyHead',
                            choices=cls.supervised_heads.keys())
        parser.add_argument('--beta',
                            help='Supervised loss weight',
                            type=float,
                            default=0.35)

        # distillation
        parser.add_argument('--distiller',
                            help='Type of distiller',
                            type=str,
                            default='BaseDistiller',
                            choices=cls.distillers.keys())
        parser.add_argument('--alpha',
                            help='Distillation weight',
                            type=float,
                            default=0.1)
        parser.add_argument('--distill-loss-fn',
                            help='Loss function for feature distillation',
                            type=str,
                            default='mse',
                            choices=['mse', 'cosine'])
        parser.add_argument('--distance-metric',
                            help='What metric to use in nmc and kmeans',
                            type=str,
                            default='L2',
                            choices=['cosine', 'L2'])
        parser.add_argument('--start-distilling-task',
                            help='Index of a task in which distillation should start',
                            type=int,
                            default=1)

        # general eval
        parser.add_argument("--eval-taw-lp", action='store_true')
        parser.add_argument("--eval-taw-knn", action='store_true')
        parser.add_argument("--eval-tag-lp", action='store_true')
        parser.add_argument("--eval-tag-knn", action='store_true')

        # linear probing
        parser.add_argument('--lp-lr',
                            help='linear probing learning rate',
                            type=float,
                            default=1e-1)
        parser.add_argument('--lp-epochs',
                            help='number of epochs for linear probing',
                            type=int,
                            default=100)
        parser.add_argument('--lp-weight-decay',
                            help='weight decay for linear probing',
                            type=float,
                            default=0)
        
        # other
        parser.add_argument("--visualize-drift", action='store_true')
        parser.add_argument("--save-ckpts", action='store_true')
        parser.add_argument("--load-ckpts", action='store_true')
        parser.add_argument("--ckpts-dir",
                            help='Directory from where checkpoints will be loaded',
                            type=str,
                            default='logs/')

        return parser

    @abc.abstractmethod
    def train_feature_extractor(self, t, train_dl_dict, valid_dl_dict):
        raise NotImplementedError()

    def save_ckpt(self, t):
        torch.save({'state_dict': self.backbone.state_dict()},
                   os.path.join(self.args.log_dir, f"task{t}.ckpt"))

    def load_ckpt(self, t):
        self.old_backbone = deepcopy(self.backbone)
        state_dict = torch.load(os.path.join(self.args.ckpts_dir, f"task{t}.ckpt"), map_location='cpu')['state_dict']
        self.backbone.load_state_dict(state_dict)

    def assimilate_knowledge(self, t, train_dl_dict, valid_dl_dict):
        adapted_prototypes = {}
        old_prototypes = deepcopy(self.prototypes)
        if t > 0 and self.args.adapt_prototypes:
            dl = copy.deepcopy(train_dl_dict["all_data"])
            dl.dataset.transforms = train_dl_dict["ssl"].dataset.transforms.base_transform
            dl = self.add_dataset_to_loader(dl, self.exemplar_dataset)
            if self.args.adapt_prototypes_algo == "ours":
                adapted_prototypes = self.adapt_centroids_ours(t, dl)
            else:
                adapted_prototypes = self.adapt_centroids_sdc(t, dl)
            self.prototypes = adapted_prototypes

        self.prototypes[t] = {}
        self.find_centroids(t, train_dl_dict, valid_dl_dict["known_class_labeled"].dataset.transforms)

        return old_prototypes, adapted_prototypes

    def adapt_centroids_ours(self, t, train_dl):
        logging.info("Training feature adaptation network")
        adapter = torch.nn.Linear(self.num_features, self.num_features).to(self.device)
        params = adapter.parameters()
        optimizer, lr_scheduler = self.get_adapter_optimizer(params)

        self.backbone.eval()
        self.old_backbone.eval()
        adapter.train()
        for epoch in tqdm(range(self.args.adapt_prototypes_epochs)):
            train_loss_mean_metric = MeanMetric()

            # not using labels, just want to have all the task data in one dl
            for images, _ in train_dl:
                images = images.to(self.device)
                optimizer.zero_grad()

                with torch.no_grad():
                    new_features = self.backbone(images)
                    old_features = self.old_backbone(images)
                old_features_translated = adapter(old_features)

                loss = torch.nn.functional.mse_loss(old_features_translated, new_features)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.args.clip)
                optimizer.step()

                train_loss_mean_metric.update(float(loss))

            logging.info(f"Epoch: {epoch}; Adaptator MSE: {train_loss_mean_metric.compute():.3f} ")

            lr_scheduler.step()

        adapter.eval()
        with torch.no_grad():
            logging.info("Calculating adapted prototypes")
            adapted_prototypes = {}
            for _t, _prototypes in self.prototypes.items():
                adapted_task_prototypes = {}
                for _c, _prototype in _prototypes.items():
                    adapted_task_prototypes[_c] = adapter(_prototype.unsqueeze(0)).squeeze()
                    logging.debug(f"Estimated drift of prototype of class {_c} from task {_t}: "
                                  f"{torch.norm(adapted_task_prototypes[_c] - _prototype):.3f}")
                adapted_prototypes[_t] = adapted_task_prototypes

        if self.args.save_ckpts:
            torch.save({'state_dict': adapter.state_dict()}, os.path.join(self.args.log_dir, f"adapter_{t}.ckpt"))
        
        return adapted_prototypes

    @torch.no_grad()
    def adapt_centroids_sdc(self, t, train_dl):
        new_features = []
        old_features = []
        self.backbone.eval()
        self.old_backbone.eval()
        for images, _ in train_dl:
            images = images.to(self.device)
            new_features.append(self.backbone(images).cpu())
            old_features.append(self.old_backbone(images).cpu())

        new_features = torch.cat(new_features)
        old_features = torch.cat(old_features)

        sigma = 0.3
        DY = new_features - old_features
        centroids = []
        for _, task_centroids in self.prototypes.items():
            centroids.extend(list(task_centroids.values()))
        centroids = torch.stack(centroids).cpu()
        distance = np.sum((np.tile(old_features[None, :, :],[centroids.shape[0], 1, 1])-np.tile(centroids[:, None, :], [1, old_features.shape[0], 1]))**2, axis=2)
        W = np.exp(-distance/(2*sigma ** 2)) + 1e-5
        W_norm = W/np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
        displacement = np.sum(np.tile(W_norm[:, :, None], [1, 1, DY.shape[1]])*np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
        displacement = torch.tensor(displacement, device=self.device)
        adapted_centroids = {}
        for k, v in self.prototypes.items():
            adapted_centroids[k] = {}
            for k2 in v.keys():
                adapted_centroids[k][k2] = self.prototypes[k][k2] + displacement[k2]

        return adapted_centroids

    def nview_forward(self, network: torch.nn.Module, x: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(x, torch.Tensor):
            return network(x.to(self.device, non_blocking=True))
        elif isinstance(x, list):
            no_chunks = len(x)
            x = torch.cat(x)
            x = network(x.to(self.device))
            return list(torch.chunk(x, no_chunks))
        else:
            raise NotImplementedError()

    def eval_task(self, train_dl_dict, valid_dl_dict, t, old_prototypes, adapted_prototypes):
        """ Eval the approach task agnostic and task aware"""
        ### Classification accuracy ###

        results = {"tag_acc": {}, "taw_acc": {}, "tag_known_acc": {}, "taw_known_acc": {}, "tag_novel_acc": {}, "taw_novel_acc": {}}
        for i in range(t + 1):
            results["tag_acc"][i], results["taw_acc"][i] = self.predict_acc_nmc(self.prototypes, valid_dl_dict[i]["all_foreground"], i)
            results["tag_known_acc"][i], results["taw_known_acc"][i] = self.predict_acc_nmc(self.prototypes, valid_dl_dict[i]["known_class"], i)
            results["tag_novel_acc"][i], results["taw_novel_acc"][i] = self.predict_acc_nmc(self.prototypes, valid_dl_dict[i]["novel_class"], i)

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

        if self.args.adapt_prototypes and t > 0:
            prev_known_dl = get_all_tasks_dataloader(train_dl_dict, 'known_class', t-1)
            prev_novel_dl = get_all_tasks_dataloader(train_dl_dict, 'novel_class', t-1)
            prev_known_classes = list(torch.unique(prev_known_dl.dataset.targets))
            prev_novel_classes = list(torch.unique(prev_novel_dl.dataset.targets))

            old_gt_known, adapted_gt_known, old_gt_novel, adapted_gt_novel = [], [], [], []

            for _t, task_protos in old_prototypes.items():
                for c, old_proto in task_protos.items():
                    _old_to_gt = torch.norm(old_proto - all_foreground_prototypes[0][c])
                    _adapted_to_gt = torch.norm(adapted_prototypes[_t][c] - all_foreground_prototypes[0][c])

                    logging.info(f"Class {c}: dist old-gt {_old_to_gt:.3f}, dist adapted-gt {_adapted_to_gt:.3f}")

                    if c in prev_known_classes:
                        old_gt_known.append(_old_to_gt.item())
                        adapted_gt_known.append(_adapted_to_gt.item())
                    elif c in prev_novel_classes:
                        old_gt_novel.append(_old_to_gt.item())
                        adapted_gt_novel.append(_adapted_to_gt.item())
                    else:
                        raise RuntimeError(f"Class {c} not in known classes {prev_known_classes} nor in novel classes {prev_novel_classes}")

            logging.info(f"Known classes: avg dist old-gt {torch.tensor(old_gt_known).mean().item():.3f}, "
                         f"dist adapted-gt {torch.tensor(adapted_gt_known).mean().item():.3f}")
            logging.info(f"Novel classes: avg dist old-gt {torch.tensor(old_gt_novel).mean().item():.3f}, "
                         f"dist adapted-gt {torch.tensor(adapted_gt_novel).mean().item():.3f}")
            logging.info(f"All classes: avg dist old-gt {torch.tensor(old_gt_known + old_gt_novel).mean().item():.3f}, "
                         f"dist adapted-gt {torch.tensor(adapted_gt_known + adapted_gt_novel).mean().item():.3f}")

            if self.args.visualize_drift:
                old_dl = get_all_tasks_dataloader(valid_dl_dict, 'all_foreground', t-1)
                new_dl = valid_dl_dict[t]["all_foreground"]
                logging.info(f"Visualizing drift")
                visualize_drift(old_dl, new_dl, self.old_backbone, self.backbone,
                                old_prototypes, adapted_prototypes, all_foreground_prototypes[0],
                                self.device, self.args.log_dir)

        ### Representation strength eval ###
        # TAw #
        # if self.args.eval_taw_lp:
        #     logging.info('Running linear probing evaluation on current task')
        #     results['taw_lp_acc'] = self.eval_linear_probing(train_dl_dict[t]['all_foreground'], valid_dl_dict[t]['all_foreground'])
        #     logging.debug('Linear probing evaluation on current task finished')
        #
        # if self.args.eval_taw_knn:
        #     logging.info('Running KNN evaluation on current task')
        #     results['taw_knn_acc'] = self.eval_knn(train_dl_dict[t]['all_foreground'], valid_dl_dict[t]['all_foreground'])
        #     logging.debug('KNN on current task finished')
        #
        # # TAg #
        # if self.args.eval_tag_lp or self.args.eval_tag_knn:
        #     all_foreground_train_dl = get_all_tasks_dataloader(train_dl_dict, 'all_foreground')
        #     all_foreground_valid_dl = get_all_tasks_dataloader(valid_dl_dict, 'all_foreground')
        #
        # if self.args.eval_tag_lp:
        #     logging.info('Running linear probing evaluation on all tasks')
        #     results['tag_lp_acc'] = self.eval_linear_probing(all_foreground_train_dl, all_foreground_valid_dl)
        #     logging.debug('Linear probing evaluation on all tasks finished')
        #
        # if self.args.eval_tag_knn:
        #     logging.info('Running KNN evaluation on all tasks')
        #     results['tag_knn_acc'] = self.eval_knn(all_foreground_train_dl, all_foreground_valid_dl)
        #     logging.debug('KNN on all tasks finished')

        return results

    def eval_linear_probing(self, train_dl, valid_dl):
        self.backbone.eval()

        num_classes = torch.unique(train_dl.dataset.targets).numel()
        lp = torch.nn.Linear(self.backbone.inplanes, num_classes, device=self.device)
        logging.debug(f"Linear probe: {lp}")

        params = list(self.backbone.parameters()) + list(lp.parameters())
        logging.debug(f'The model has {sum(p.numel() for p in params if p.requires_grad):,} trainable parameters')
        logging.debug(f'The model has {sum(p.numel() for p in params if not p.requires_grad):,} frozen parameters')

        optimizer, lr_scheduler = self.get_lp_optimizer(lp.parameters())

        # map global targets to local targets
        train_dl = deepcopy(train_dl)
        valid_dl = deepcopy(valid_dl)
        global_labels = train_dl.dataset.targets.unique()
        global_to_task_label_dict = {k.item(): i for i, k in enumerate(global_labels)}
        train_dl.dataset.targets.apply_(lambda x: global_to_task_label_dict[x])
        valid_dl.dataset.targets.apply_(lambda x: global_to_task_label_dict[x])

        for epoch in range(self.lp_epochs):
            metrics = {
                "train_loss": MeanMetric(),
                "train_acc": Accuracy("multiclass", num_classes=num_classes),
                "val_loss": MeanMetric(),
                "val_acc": Accuracy("multiclass", num_classes=num_classes)
            }

            for images, target in train_dl:
                images, target = images.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    features = self.backbone(images)
                logits = lp(features)
                loss = torch.nn.functional.cross_entropy(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.lp_clip)
                optimizer.step()
                metrics["train_loss"].update(float(loss))
                metrics["train_acc"].update(torch.argmax(logits, dim=1).detach().cpu(), target.cpu())

            lr_scheduler.step()

            with torch.no_grad():
                for images, target in valid_dl:
                    images, target = images.to(self.device), target.to(self.device)
                    features = self.backbone(images)
                    logits = lp(features)
                    loss = torch.nn.functional.cross_entropy(logits, target)
                    metrics["val_loss"].update(float(loss))
                    metrics["val_acc"].update(torch.argmax(logits, dim=1).cpu(), target.cpu())

            s = f"Epoch: {epoch} "
            for k, v in metrics.items():
                s += f"{k}: {v.compute():.2f} "
            logging.debug(s)
        logging.info(s)
        
        return metrics['val_acc'].compute()

    @torch.no_grad()
    def eval_knn(self, train_dl, valid_dl):
        self.backbone.eval()
        knn = KNN()

        # map global targets to local targets
        train_dl = deepcopy(train_dl)
        valid_dl = deepcopy(valid_dl)
        train_dl.dataset.transforms = valid_dl.dataset.transforms
        global_labels = train_dl.dataset.targets.unique()
        global_to_task_label_dict = {k.item(): i for i, k in enumerate(global_labels)}
        train_dl.dataset.targets.apply_(lambda x: global_to_task_label_dict[x])
        valid_dl.dataset.targets.apply_(lambda x: global_to_task_label_dict[x])

        train_features, train_targets, valid_features, valid_targets = [], [], [], []

        for images, target in train_dl:
            images, target = images.to(self.device), target.to(self.device)
            features = self.backbone(images)
            train_features.append(features)
            train_targets.append(target)

        for images, target in valid_dl:
            images, target = images.to(self.device), target.to(self.device)
            features = self.backbone(images)
            valid_features.append(features)
            valid_targets.append(target)

        train_features = torch.cat(train_features)
        train_targets = torch.cat(train_targets)
        valid_features = torch.cat(valid_features)
        valid_targets = torch.cat(valid_targets)

        acc1, acc5 = knn.compute(train_features, train_targets, valid_features, valid_targets)

        logging.info(f"KNN evaluation acc1: {acc1}, acc5: {acc5}")

        return acc1

    def get_optimizer(self, parameters, weight_decay=0.):
        if self.optimizer == "SGD":
            optimizer = SGD(parameters, lr=self.lr, weight_decay=weight_decay, momentum=0.9)
        elif self.optimizer == "AdamW":
            optimizer = AdamW(parameters, lr=self.lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError()

        if self.lr_scheduler == "ExponentialLR":
            scheduler = WarmUpExponentialLR(optimizer, self.lr_decay, self.warmup_iters)
        elif self.lr_scheduler == "CosineAnnealingLR":
            scheduler = WarmUpCosineAnnealingLR(optimizer, self.epochs, self.lr * 1e-3, self.warmup_iters)
        else:
            raise NotImplementedError()
        
        return optimizer, scheduler

    def get_adapter_optimizer(self, parameters):
        optimizer = SGD(parameters, lr=self.lp_lr, weight_decay=1e-5, momentum=0.9)
        scheduler = ExponentialLR(optimizer, 0.95)
        return optimizer, scheduler

    @torch.no_grad()
    def calculate_prototypes(self, trn_loader, test_transforms):
        """ Fill prototypes_dict dict based on prototypes of labeled data """
        prototypes_dict = {}
        classes_labeled = np.unique(trn_loader.dataset.targets)
        # Calculate prototypes for labeled data
        self.backbone.eval()
        for c in classes_labeled:
            is_c = torch.tensor(trn_loader.dataset.targets) == c
            ds = trn_loader.dataset.images[is_c]
            ds = MemoryDataset(ds, trn_loader.dataset.targets[is_c], test_transforms)
            loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)
            from_ = 0
            class_features = torch.full((len(ds), self.num_features), fill_value=-999., device=self.device)
            for images, _ in loader:
                bsz = images.shape[0]
                images = images.to(self.device)
                features = self.backbone(images)
                class_features[from_: from_ + bsz] = features
                from_ += bsz
            prototype = torch.mean(class_features, dim=0)
            prototypes_dict[c] = prototype
        return prototypes_dict

    @torch.no_grad()
    def find_centroids_standard(self, t, train_dl_dict, test_transforms):
        """ Calculate centroids based on all data available in the task """
        images = train_dl_dict["all_data"].dataset.images
        ds = MemoryDataset(images, np.zeros_like(images[:, 0, 0, 0]), test_transforms)
        loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=train_dl_dict["novel_class"].num_workers, shuffle=False)
        from_ = 0
        all_features = torch.full((len(ds), self.num_features), fill_value=-999., device=self.device)
        for images, _ in loader:
            bsz = images.shape[0]
            images = images.to(self.device)
            features = self.backbone(images)
            all_features[from_: from_ + bsz] = features
            from_ += bsz

        total_centroids = self.classes_per_task #TODO: How should we predict this?
        if self.clustering_strategy == "KMeans":
            centroids = KMeans(all_features, k=total_centroids, n_iters=100, distance_metric=self.args.distance_metric).train_pts
        else:  # KMedians
            centroids = KMedians(all_features, k=total_centroids, n_iters=100, distance_metric=self.args.distance_metric).train_pts
        prototypes = torch.stack([p for p in self.prototypes[t].values()])

        # Assign prototypes (labeled data) to centroids (unlabeled data), find novel classes
        # http://www.assignmentproblems.com/doc/LSAPIntroduction.pdf
        cost_matrix = np.array(torch.cdist(prototypes, centroids).cpu())
        assignment = linear_sum_assignment(cost_matrix)
        # if self.args.prototype_update:
        #     for i, c in enumerate(assignment[0]):
        #         self.prototypes[t][c] = centroids[assignment[1][i]]

        # Make unassigned centroids to be prototypes of novel classes. Take label based on the best possible assignment with ground truth
        centroids_left = set([_ for _ in range(total_centroids)]).difference(set(assignment[1]))
        centroids = centroids[list(centroids_left)]
        protos_dict = self.calculate_prototypes(train_dl_dict["novel_class"], test_transforms)
        idx_to_proto = list(protos_dict.keys())
        prototypes = torch.stack(list(protos_dict.values()))
        cost_matrix = np.array(torch.cdist(prototypes, centroids).cpu())
        assignment = linear_sum_assignment(cost_matrix)
        for i, c in enumerate(assignment[0]):
            self.prototypes[t][idx_to_proto[c]] = centroids[assignment[1][i]]

    @torch.no_grad()
    def find_centroids_semi_sup(self, t, train_dl_dict, test_transforms):
        """ Calculate centroids using semi supervised kmeans as in GCD """
        images = [train_dl_dict["known_class_unlabeled"].dataset.images, train_dl_dict["known_class_labeled"].dataset.images,\
              train_dl_dict["novel_class"].dataset.images]
        labels = [train_dl_dict["known_class_unlabeled"].dataset.targets, train_dl_dict["known_class_labeled"].dataset.targets,\
              train_dl_dict["novel_class"].dataset.targets]
        images = np.concatenate(images)
        labels = np.concatenate(labels)
        mask_is_lab = np.zeros((labels.shape[0]), dtype=bool)
        from_ = train_dl_dict["known_class_unlabeled"].dataset.targets.shape[0]
        to_ = from_ + train_dl_dict["known_class_labeled"].dataset.targets.shape[0]
        mask_is_lab[from_:to_] = True
        ds = MemoryDataset(images, labels, test_transforms)
        loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=train_dl_dict["novel_class"].num_workers, shuffle=False)
        from_ = 0
        all_features = torch.full((len(ds), self.num_features), fill_value=-999., device=self.device)
        all_labels = torch.full((len(ds),), fill_value=-999., device=self.device)
        for images, labels in loader:
            bsz = images.shape[0]
            images = images.to(self.device)
            features = self.backbone(images)
            all_features[from_: from_ + bsz] = features
            all_labels[from_: from_ + bsz] = labels
            from_ += bsz

        all_features = all_features.cpu().numpy()
        all_labels = all_labels.cpu().numpy()

        l_feats = all_features[mask_is_lab]  # Get labelled set
        u_feats = all_features[~mask_is_lab]  # Get unlabelled set
        l_targets = all_labels[mask_is_lab]  # Get labelled targets
        u_targets = all_labels[~mask_is_lab]  # Get unlabelled targets

        logging.info('Fitting Semi-Supervised K-Means...')
        if self.clustering_strategy == "SemiSupKMeans":
            kmeans = SemiSupKMeans(k=self.classes_per_task, tolerance=1e-4, max_iterations=200, init='k-means++',
                                   n_init=100, random_state=None, n_jobs=None, pairwise_batch_size=1024, mode=None)
        else:
            kmeans = SemiSupKMedians(k=self.classes_per_task, tolerance=1e-4, max_iterations=200, init='k-means++',
                                     n_init=100, random_state=None, n_jobs=None, pairwise_batch_size=1024, mode=None)

        l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(self.device) for
                                                  x in (l_feats, u_feats, l_targets, u_targets))

        kmeans.fit_mix(u_feats, l_feats, l_targets)
        # all_preds = kmeans.labels_.cpu().numpy()
        # u_targets = u_targets.cpu().numpy()
        centroids = kmeans.cluster_centers_

        for i in range(centroids.shape[0]):
            self.prototypes[t][i+t*self.classes_per_task] = centroids[i]

        novel_centroids = centroids[-self.novel_classes_per_task:]
        protos_dict = self.calculate_prototypes(train_dl_dict["novel_class"], test_transforms)
        idx_to_proto = list(protos_dict.keys())
        prototypes = torch.stack(list(protos_dict.values()))
        cost_matrix = np.array(torch.cdist(prototypes, novel_centroids).cpu())
        assignment = linear_sum_assignment(cost_matrix)
        for i, c in enumerate(assignment[0]):
            self.prototypes[t][idx_to_proto[c]] = novel_centroids[assignment[1][i]]

    @torch.no_grad()
    def predict_acc_nmc(self, prototypes_dict, loader, t, offset=0):
        """ Perform nearest mean classification based on distance to prototypes """
        self.backbone.eval()
        prototypes = []
        for j in prototypes_dict.keys():
            prototypes.extend(list(prototypes_dict[j].values()))
        prototypes = torch.stack(prototypes)
        tag_acc = Accuracy("multiclass", num_classes=prototypes.shape[0])
        taw_acc = Accuracy("multiclass", num_classes=self.classes_per_task)
        taw_offset = t * self.classes_per_task
        for images, target in loader:
            images = images.to(self.device)
            features = self.backbone(images)
            if self.args.distance_metric == "L2":
                dist = torch.cdist(features, prototypes)
                tag_preds = torch.argmin(dist, dim=1)
                taw_preds = torch.argmin(dist[:, taw_offset: taw_offset + self.classes_per_task], dim=1) + taw_offset
            else: # cosine
                cos_sim = F.normalize(features) @ F.normalize(prototypes).T
                tag_preds = torch.argmax(cos_sim, dim=1)
                taw_preds = torch.argmax(cos_sim[:, taw_offset: taw_offset + self.classes_per_task], dim=1) + taw_offset

            tag_acc.update(tag_preds.cpu(), target)
            taw_acc.update(taw_preds.cpu(), target)

        return float(tag_acc.compute()), float(taw_acc.compute())

    @torch.no_grad()
    def store_exemplars(self, t, train_dl_dict):
        """ Calculate centroids using semi supervised kmeans as in GCD """
        images = train_dl_dict["known_class_labeled"].dataset.images
        labels = train_dl_dict["known_class_labeled"].dataset.targets
        if self.exemplar_dataset is None:
            self.exemplar_dataset = MemoryDataset(np.zeros((0, *images.shape[1:]), dtype=np.uint8), np.zeros((0,), dtype=np.int64), train_dl_dict["known_class_labeled"].dataset.transforms)
        classes = torch.unique(labels)
        for c in classes:
            class_images = images[labels == c]
            indices = list(range(class_images.shape[0]))
            random.shuffle(indices)
            indices = indices[:self.num_exemplars]
            exemplar_images = class_images[indices]
            exemplar_labels = np.full((exemplar_images.shape[0],), fill_value=c, dtype=np.int64)
            self.exemplar_dataset.images = np.concatenate((self.exemplar_dataset.images, exemplar_images), axis=0)
            self.exemplar_dataset.targets = np.concatenate((self.exemplar_dataset.targets, exemplar_labels), axis=0)

    @torch.no_grad()
    def add_dataset_to_loader(self, dl, ds):
        if ds is None:
            return dl
        dl.dataset.images = np.concatenate((dl.dataset.images, ds.images), axis=0)
        dl.dataset.targets = np.concatenate((dl.dataset.targets, ds.targets), axis=0)
        return dl
