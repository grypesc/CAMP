import copy
import numpy as np

from torchvision import transforms
from torchvision.datasets import StanfordCars as StanfordCarsTV

from datasets.base import BaseDataModule, MemoryDataset, SSLMemoryDataset, ContrastiveLearningViewGenerator


class StanfordCars(BaseDataModule):
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

    def __init__(self, args):
        super().__init__(args)
        self.total_classes = 196
        self.create_datasets(args.data_dir, args.num_tasks, args.fraction_novel, args.fraction_labeled, args.fraction_bg)
        self.classes_per_task = self.total_classes // args.num_tasks

    def load_data(self, data_dir, split):
        _dataset = StanfordCarsTV(data_dir, split=split, download=True)
        images, targets = [], []
        for img, target in _dataset:
            images.append(np.array(img.resize(self.image_size)))
            targets.append(target)
        images = np.stack(images)
        targets = np.stack(targets)
        return MemoryDataset(images, targets, None)
        
    def create_datasets(self, data_dir, num_tasks, fraction_novel=0.2, fraction_labeled=0.2, fraction_bg=0.05):
        """Produces train and val datasets"""
        trn_dataset = self.load_data(data_dir=data_dir, split='train')
        val_dataset = self.load_data(data_dir=data_dir, split='test')

        class_order = [186,  41,  73, 102, 194,  23, 180, 144, 173,  71, 135,  10, 182, 147,
                        172,  83, 120, 132, 105,  26,  92, 122,  77, 131,  68,  95, 100, 125,
                        6,  57,  76,  21, 106,  53, 161,  40, 189, 156,   1,  97, 176,  25,
                        37,  78,  49,  45,  56,  99,  51,  33,  72, 153,  65, 167, 178, 134,
                        61,  91, 145, 126,  62, 183,  12, 190, 141, 177, 163, 150,  13, 139,
                        117, 108,  15,  34, 142,  89,  81,  52,  80,  67,  50,   0, 188, 119,
                        2,  17,   4,  30,   5,  63, 121,  32,  75, 128, 103,  59,  66, 152,
                        179,  64,  70,  39, 162,  31, 133,  87,  38,  88,  94, 110, 164, 107,
                        175, 174,  96, 149,  14,  29, 127,  35,  22, 136,  69, 158, 193, 101,
                        16, 191, 154,  82, 192, 157,  24,  47, 137,   9,  48,  28,  74, 114,
                        86, 171, 143, 146, 187, 168, 138,  93, 123,  90,  58, 151, 113, 129,
                        159,  27, 195, 130,  18,  85, 118, 140, 115,  79,  20,  60, 109,  43,
                        54,  46, 116, 181,  11, 170,  84, 148,  55, 166,   7, 111,  19, 155,
                        42,  98, 160, 184, 165,  36, 185,   8, 169, 104, 124,   3, 112,  44]
        self.original_class_order = class_order

        # Map labels to 0, 1, 2 ...
        trn_targets = copy.deepcopy(trn_dataset.targets)
        val_targets = copy.deepcopy(val_dataset.targets)
        for i, c in enumerate(class_order):
            trn_dataset.targets[trn_targets == c] = i
            val_dataset.targets[val_targets == c] = i
        class_order = list(range(len(class_order)))

        self.novel_classes_per_task = int(len(class_order) * fraction_novel // num_tasks)
        task_to_classes = np.array_split(class_order, num_tasks)

        task_to_novel_class = [t[-self.novel_classes_per_task:] for t in task_to_classes]
        task_to_known_class = [t[:-self.novel_classes_per_task] for t in task_to_classes]
        ssl_transforms = ContrastiveLearningViewGenerator(self.ssl_transform, 2)

        self.train_dataset_dict = self._prepare_dataset(
            trn_dataset, task_to_novel_class, task_to_known_class, self.sl_transform, ssl_transforms, fraction_labeled, fraction_bg)

        self.val_dataset_dict = self._prepare_dataset(
            val_dataset, task_to_novel_class, task_to_known_class, self.test_transform, ssl_transforms, fraction_labeled, fraction_bg)


class StanfordCarsx224(StanfordCars):
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

