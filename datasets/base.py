import abc
import copy
import os
import random

import numpy as np
import torch
from typing import Dict

from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Tuple
from PIL import Image


class ContrastiveLearningViewGenerator(object):
    """Generates n views of the same image"""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]


class MemoryDataset(torch.utils.data.Dataset):
    def __init__(self, images, targets, transforms, map_classes=False, offset=0):
        super().__init__()
        self.images = images
        self.data = images
        self.targets = targets
        self.transforms = transforms

        if map_classes:
            for idx, c in enumerate(sorted(list((np.unique(targets))))):
                self.targets[self.targets == c] = idx + offset

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        image = self.transforms(image)
        return image, self.targets[index]

    def get_double_view(self):
        pass


class SSLMemoryDataset(torch.utils.data.Dataset): # TODO: Merge with ContrastiveLearningViewGenerator
    def __init__(self, images: list, transforms):
        super().__init__()
        self.images = np.concatenate(images)
        self.transforms = transforms

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        image = self.transforms(image)
        return image


class BaseDataModule(abc.ABC, Dataset):
    image_size: Tuple
    ssl_transform: transforms.Compose
    sl_transform: transforms.Compose
    test_transform: transforms.Compose

    def __init__(self, args):
        super().__init__()

        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.train_dataset_dict = {}
        self.val_dataset_dict = {}
        self.test_dataset_dict = {}
        self.total_classes = None
        self.classes_per_task = None
        self.novel_classes_per_task = None

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser.add_argument('--data-dir',
                            help='dataset location',
                            type=str,
                            required=True)
        parser.add_argument('--batch-size',
                            help='batch size',
                            type=int,
                            default=64)
        parser.add_argument('--num-tasks',
                            help='number of tasks',
                            type=int,
                            default=10)
        parser.add_argument('--fraction-novel',
                            help='fraction of novel classes',
                            type=float,
                            default=0.2)
        parser.add_argument('--fraction-labeled',
                            help='fraction of known data that will be labelled',
                            type=float,
                            default=0.5)
        parser.add_argument('--fraction-bg',
                            help='fraction of background/noise datasamples',
                            type=float,
                            default=0.0)
        parser.add_argument('--num-workers',
                            help='dataloader workers per DDP process',
                            type=int,
                            default=0)
        parser.add_argument('--n_views',
                            help='number of views for contrastive loss',
                            default=2,
                            type=int)
        return parser

    def _prepare_dataset(self, dataset, task_to_novel_class, task_to_known_class, transforms, ssl_transforms, fraction_labeled, fraction_bg):
        """ Each dataset is a dict of tasks. Each task is a dict of 6 datasets."""
        num_tasks = len(task_to_novel_class)

        # All foreground, labeled
        foreground_dict = {}
        for t in range(num_tasks):
            images, targets = [], []
            classes_in_t = list(task_to_known_class[t]) + list(task_to_novel_class[t])
            for c in classes_in_t:
                is_datasample = dataset.targets == c
                samples = dataset.data[is_datasample]
                images.append(samples)
                targets.append(torch.full((samples.shape[0],), fill_value=c))
            images = np.concatenate(images)
            target = torch.cat(targets)
            foreground_dict[t] = MemoryDataset(images, target, transforms=transforms)

        # Background dataset, labels are -1
        # bg_img_size = images.shape[1:3]
        # bg_images, bg_labels = self.load_background_dataset(self.data_dir, int(len(dataset.targets) * fraction_bg), bg_img_size)
        # bg_dataset_dict = {}
        # for t in range(num_tasks):
        #     from_ = int(t * len(bg_images) / num_tasks)
        #     to_ = int((t+1) * len(bg_images) / num_tasks)
        #     bg_dataset_dict[t] = MemoryDataset(bg_images[from_:to_], bg_labels[from_:to_], ssl_transforms)

        # All data, labeled
        all_data_dict = {}
        for t in range(num_tasks):
            images = foreground_dict[t].images
            targets = foreground_dict[t].targets
            # images = np.concatenate((foreground_dict[t].images, bg_dataset_dict[t].images), axis=0)
            # targets = np.concatenate((foreground_dict[t].targets, bg_dataset_dict[t].targets), axis=0)
            all_data_dict[t] = MemoryDataset(images, targets, transforms=transforms)

        # Novel classes, unlabeled
        novel_class_dataset_dict = {}
        for t in range(num_tasks):
            images, target = [], []
            novel_classes_in_t = list(task_to_novel_class[t])
            for c in novel_classes_in_t:
                is_datasample = dataset.targets == c
                samples = dataset.data[is_datasample]
                images.append(samples)
                target.append(torch.full((samples.shape[0],), fill_value=c))
            images = np.concatenate(images)
            target = torch.cat(target)
            novel_class_dataset_dict[t] = MemoryDataset(images, target, transforms=transforms, offset=len(list(task_to_known_class[t])))

        # Known classes, labeled and unlabeled
        known_classes_dataset_dict, known_class_labeled_dataset_dict, known_class_unlabeled_dataset_dict = {}, {}, {}
        for t in range(num_tasks):
            images_labeled, images_unlabeled, target_labeled, target_unlabeled = [], [], [], []
            known_classes_in_t = list(task_to_known_class[t])
            for c in known_classes_in_t:
                is_datasample = dataset.targets == c
                samples = dataset.data[is_datasample]
                is_labeled = np.random.random((samples.shape[0],) ) < fraction_labeled
                images_labeled.append(samples[is_labeled])
                images_unlabeled.append(samples[~is_labeled])
                target_labeled.append(torch.full((is_labeled.sum(),), fill_value=c))
                target_unlabeled.append(torch.full(((~is_labeled).sum(),), fill_value=c))
            images_labeled = np.concatenate(images_labeled)
            images_unlabeled = np.concatenate(images_unlabeled)
            target_labeled = torch.cat(target_labeled)
            target_unlabeled = torch.cat(target_unlabeled)
            known_classes_dataset_dict[t] = MemoryDataset(np.concatenate((images_labeled, images_unlabeled)), torch.cat((target_labeled, target_unlabeled)), transforms=transforms)
            known_class_labeled_dataset_dict[t] = MemoryDataset(images_labeled, target_labeled, transforms=transforms)
            known_class_unlabeled_dataset_dict[t] = MemoryDataset(images_unlabeled, target_unlabeled, transforms=transforms)

        # SSL dataset consisting of all samples in a task, unlabeled, perfect for SSL forward pass and SSL loss
        ssl_dataset = {}
        for t in range(num_tasks):
            ssl_dataset[t] = SSLMemoryDataset([all_data_dict[t].images], ssl_transforms)

        full_dataset = {}
        for t in range(num_tasks):
            full_dataset[t] = {
                "all_data": all_data_dict[t],
                "all_foreground": foreground_dict[t],
                # "background": bg_dataset_dict[t],
                "novel_class": novel_class_dataset_dict[t],
                "known_class": known_classes_dataset_dict[t],
                "known_class_labeled": known_class_labeled_dataset_dict[t],
                "known_class_unlabeled": known_class_unlabeled_dataset_dict[t],
                "ssl": ssl_dataset[t]
            }
        return full_dataset

    def load_background_dataset(self, data_dir, num_images, img_size):
        """Load background images from imagenet subset and set their labels to -1"""
        if num_images == 0:
            return np.empty((0, *img_size, 3), dtype=np.uint8), np.empty((0,), dtype=np.uint8)
        path_to_imagenet_train = os.path.join(data_dir, "seed_1993_subset_100_imagenet", "data", "train")
        image_paths = []
        for class_dir in os.listdir(path_to_imagenet_train):
            images = os.listdir(os.path.join(path_to_imagenet_train, class_dir))
            images = list(map(lambda x: os.path.join(class_dir, x), images))
            image_paths.extend(images)
        images = []
        image_paths = random.sample(image_paths, num_images)
        for path in image_paths:
            image = Image.open(os.path.join(path_to_imagenet_train, path)).convert('RGB').resize(img_size)
            images.append(image)
        return np.stack(images), np.full((len(images),), fill_value=-1)

    @abc.abstractmethod
    def create_datasets(self, *args, **kwargs):
        raise NotImplementedError("Life is a bitch.")

    def create_dataloaders(self, dataset_dict):
        dataloader_dict = {}
        for task_id, datasets in dataset_dict.items():
            task_dataloaders = {}
            for key, dataset in datasets.items():
                task_dataloaders[key] = DataLoader(dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True,
                                                   pin_memory=True, persistent_workers=False) \
                    if len(dataset) > 0 else EmptyDataLoader(dataset)
            dataloader_dict[task_id] = task_dataloaders
        return dataloader_dict

    def train_dataloader(self):
        return self.create_dataloaders(self.train_dataset_dict)

    def val_dataloader(self):
        return self.create_dataloaders(self.val_dataset_dict)

    def test_dataloader(self):
        return self.create_dataloaders(self.test_dataset_dict)

    def __str__(self):
        s = "\n"
        for ds_name, ds in [("Train", self.train_dataset_dict), ("Valid", self.val_dataset_dict), ("Test", self.test_dataset_dict)]:
            if not ds:
                continue
            s += f"{ds_name} dataset info: \n"
            for task_id, datasets in ds.items():
                s += f"Task: {task_id}, number of samples: \n"
                for key, dataset in datasets.items():
                    s += f"\t{key}: {len(dataset)} \n"
        return s


class DataLoaderCyclicIterator:
    def __init__(self, dl) -> None:
        self.dl = dl
        self.iterator = iter(self.dl)
        self.next_iterator = iter(self.dl)
        
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = self.next_iterator
            self.next_iterator = iter(self.dl)
            return next(self.iterator)


class EmptyDataLoader(torch.nn.Module):
    """ Dummy dataloader for empty datasets """
    def __init__(self, dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, num_workers=0,
                 collate_fn=None, pin_memory=False):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size


def get_all_tasks_dataloader(dl_task_dict: Dict, key: str, up_to_task: int = None):
    if up_to_task is None:
        up_to_task = max(dl_task_dict.keys())

    _dl_task_dict = {k: v for k, v in dl_task_dict.items() if k in range(up_to_task+1)}

    images = [dl_dict[key].dataset.images for dl_dict in _dl_task_dict.values()]
    targets = [copy.deepcopy(dl_dict[key].dataset.targets) for dl_dict in _dl_task_dict.values()]
    images = np.concatenate(images)
    targets = torch.cat(targets)
    transforms = _dl_task_dict[0][key].dataset.transforms

    all_tasks_ds = MemoryDataset(images, targets, transforms=transforms)
    all_tasks_dl = DataLoader(all_tasks_ds,
                              num_workers=_dl_task_dict[0][key].num_workers,
                              batch_size=_dl_task_dict[0][key].batch_size,
                              shuffle=True)

    return all_tasks_dl
