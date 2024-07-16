import os
import numpy as np

from PIL import Image
from torchvision import transforms

from datasets.base import BaseDataModule, MemoryDataset, ContrastiveLearningViewGenerator


class DomainNet(BaseDataModule):
    image_size = (224, 224)
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

    def __init__(self, args, total_classes=60):
        super().__init__(args)
        self.total_classes = total_classes
        self.classes_per_task = self.total_classes // 6
        self.create_datasets(args.data_dir, 6, args.fraction_novel, args.fraction_labeled, args.fraction_bg)

    def _load_data_(self, data_dir):
        train_images, train_labels = [], []
        val_images, val_labels = [], []

        path = os.path.join(data_dir, "domainnet")
        for d, domain in enumerate(["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]):
            classes = os.listdir(os.path.join(path, domain))[:self.classes_per_task]
            for i, class_name in enumerate(classes):
                class_path = os.path.join(path, domain, class_name)
                image_names = os.listdir(class_path)
                for j, image_name in enumerate(image_names):
                    image = Image.open(os.path.join(class_path, image_name)).convert('RGB').resize(self.image_size)
                    # Put ~10% of class images in val
                    if j > 0.1 * len(image_names):
                        train_images.append(np.array(image))
                        train_labels.append(int(d * self.classes_per_task + i))
                    else:
                        val_images.append(np.array(image))
                        val_labels.append(int(d * self.classes_per_task + i))

        train_images = np.stack(train_images)
        train_labels = np.stack(train_labels)
        val_images = np.stack(val_images)
        val_labels = np.stack(val_labels)
        return MemoryDataset(train_images, train_labels, None), MemoryDataset(val_images, val_labels, None)

    def create_datasets(self, data_dir, num_tasks, fraction_novel, fraction_labeled, fraction_bg):
        """Produces train and val datasets. Tiny imagenet has different data structure for train and val, so it is a bit messy"""

        trn_dataset, val_dataset = self._load_data_(data_dir)

        self.novel_classes_per_task = int(self.total_classes * fraction_novel // num_tasks)
        task_to_classes = np.array_split(list(range(self.total_classes)), num_tasks)

        task_to_novel_class = [t[-self.novel_classes_per_task:] for t in task_to_classes]
        task_to_known_class = [t[:-self.novel_classes_per_task] for t in task_to_classes]
        ssl_transforms = ContrastiveLearningViewGenerator(self.ssl_transform, 2)

        self.train_dataset_dict = self._prepare_dataset(
            trn_dataset, task_to_novel_class, task_to_known_class, self.sl_transform, ssl_transforms, fraction_labeled, fraction_bg)

        self.val_dataset_dict = self._prepare_dataset(
            val_dataset, task_to_novel_class, task_to_known_class, self.test_transform, ssl_transforms, fraction_labeled, fraction_bg)


class DomainNetSubset(DomainNet):
    def __init__(self, args):
        super().__init__(args, total_classes=10)
