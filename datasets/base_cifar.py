import abc
import copy
import os

import numpy as np
import random
import torch

from torchvision import transforms

from datasets.base import BaseDataModule, MemoryDataset, SSLMemoryDataset, ContrastiveLearningViewGenerator


class BaseCIFAR(BaseDataModule):
    image_size = (32, 32)
    ssl_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.08, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
    sl_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.ColorJitter(0.3, 0.3, 0.1, 0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
    test_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])

    def __init__(self, args, total_classes):
        super().__init__(args)
        self.total_classes = total_classes
        self.create_datasets(args.data_dir, args.num_tasks, args.fraction_novel, args.fraction_labeled, args.fraction_bg)
        self.classes_per_task = self.total_classes // args.num_tasks

    @abc.abstractmethod
    def get_dataset_class(self):
        raise NotImplementedError()

    def create_datasets(self, data_dir, num_tasks, fraction_novel=0.2, fraction_labeled=0.2, fraction_bg=0.05):
        """Produces train and val datasets"""

        trn_dataset = self.get_dataset_class()(data_dir, train=True, download=True)
        trn_dataset.targets = np.array(trn_dataset.targets)

        val_dataset = self.get_dataset_class()(data_dir, train=False, download=True)
        val_dataset.targets = np.array(val_dataset.targets)

        # Map labels to 0, 1, 2 ...
        trn_targets = copy.deepcopy(trn_dataset.targets)
        val_targets = copy.deepcopy(val_dataset.targets)
        for i, c in enumerate(self.original_class_order):
            trn_dataset.targets[trn_targets == c] = i
            val_dataset.targets[val_targets == c] = i
        class_order = list(range(len(self.original_class_order)))

        self.novel_classes_per_task = int(len(class_order) * fraction_novel // num_tasks)
        task_to_classes = np.array_split(class_order, num_tasks)

        task_to_novel_class = [t[-self.novel_classes_per_task:] for t in task_to_classes]
        task_to_known_class = [t[:-self.novel_classes_per_task] for t in task_to_classes]
        ssl_transforms = ContrastiveLearningViewGenerator(self.ssl_transform, 2)

        self.train_dataset_dict = self._prepare_dataset(
            trn_dataset, task_to_novel_class, task_to_known_class, self.sl_transform, ssl_transforms, fraction_labeled, fraction_bg)

        self.val_dataset_dict = self._prepare_dataset(
            val_dataset, task_to_novel_class, task_to_known_class, self.test_transform, ssl_transforms, fraction_labeled, fraction_bg)


def base_cifar_224_wrapper(BaseCIFARClass=object):
    class CIFARx224(BaseCIFARClass):
        image_size = (224, 224)

        ssl_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.08, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ])
        sl_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size, padding=16),
                transforms.ColorJitter(0.3, 0.3, 0.1, 0.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ])
        test_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ])

        def __init__(self, args):
            super().__init__(args)

    return CIFARx224
