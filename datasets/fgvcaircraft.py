import copy
import numpy as np

from PIL import Image
from torchvision import transforms
from torchvision.datasets import FGVCAircraft as FGVCAircraftTV

from datasets.base import BaseDataModule, MemoryDataset, ContrastiveLearningViewGenerator


class FGVCAircraft(BaseDataModule):
    image_size = (32, 32)
    original_class_order = [68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
        28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
        98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
        36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33]
    ssl_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.3, 1.0),
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

    def __init__(self, args):
        super().__init__(args)
        self.total_classes = 100
        self.create_datasets(args.data_dir, args.num_tasks, args.fraction_novel, args.fraction_labeled, args.fraction_bg)
        self.classes_per_task = self.total_classes // args.num_tasks

    def load_data(self, dataset):
        data = []
        for path in dataset._image_files:
            image = Image.open(path).convert('RGB').resize(self.image_size)
            data.append(image)
        return np.stack(data)

    def create_datasets(self, data_dir, num_tasks, fraction_novel=0.2, fraction_labeled=0.2, fraction_bg=0.05):
        """Produces train and val datasets. Tiny imagenet has different data structure for train and val, so it is a bit messy"""

        trn_dataset = FGVCAircraftTV(data_dir, split="trainval", download=True)
        trn_dataset.targets = np.array(trn_dataset._labels)

        val_dataset = FGVCAircraftTV(data_dir, split="test", download=True)
        val_dataset.targets = np.array(val_dataset._labels)
        trn_dataset.data = self.load_data(trn_dataset)
        val_dataset.data = self.load_data(val_dataset)

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


class FGVCAircraftx224(FGVCAircraft):
    image_size = (224, 224)

    ssl_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.3, 1.0),
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
