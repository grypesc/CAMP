import os
import numpy as np

from PIL import Image
from torchvision import transforms

from datasets.base import BaseDataModule, MemoryDataset, ContrastiveLearningViewGenerator


class TinyImageNet(BaseDataModule):
    image_size = (64, 64)
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
        self.total_classes = 200
        self.create_datasets(args.data_dir, args.num_tasks, args.fraction_novel, args.fraction_labeled, args.fraction_bg)
        self.classes_per_task = self.total_classes // args.num_tasks

    def get_dataset_class(self):
        raise TinyImageNet

    def create_datasets(self, data_dir, num_tasks, fraction_novel=0.2, fraction_labeled=0.2, fraction_bg=0.05):
        """Produces train and val datasets. Tiny imagenet has different data structure for train and val, so it is a bit messy"""

        def load_data(data_dir, subset, classs_order):
            id_to_classes = {i: classs_order[i] for i in range(len(classs_order))}
            images, labels = [], []
            if subset == "train":
                path = os.path.join(data_dir, "tiny-imagenet-200", subset)
                for i, class_name in id_to_classes.items():
                    class_path = os.path.join(path, class_name, "images")
                    image_names = os.listdir(class_path)
                    for image_name in image_names:
                        image = Image.open(os.path.join(class_path, image_name)).convert('RGB')
                        images.append(np.array(image))
                        labels.append(i)
            else: # val
                classes_to_id = {v: k for k, v in id_to_classes.items()}
                path = os.path.join(data_dir, "tiny-imagenet-200", subset)
                with open(os.path.join(path, 'val_annotations.txt')) as f:
                    lines = f.readlines()
                for line in lines:
                    name, label, *_ = line.split("\t")
                    label = classes_to_id[label]
                    image = Image.open(os.path.join(path, "images",  name)).convert('RGB')
                    images.append(np.array(image))
                    labels.append(label)

            images = np.stack(images)
            targets = np.stack(labels)
            return MemoryDataset(images, targets, None)

        class_order = ['n01983481', 'n09193705', 'n03796401', 'n03649909', 'n02788148', 'n07747607', 'n03201208', 'n02124075', 'n04366367', 'n03584254', 'n01784675', 'n02963159', 'n02268443', 'n02666196', 'n03042490', 'n09428293', 'n07749582', 'n03662601', 'n07753592', 'n02802426', 'n01770393', 'n06596364', 'n01629819', 'n04507155', 'n02892201', 'n03160309', 'n04254777', 'n02988304', 'n03400231', 'n09256479', 'n04417672', 'n01950731', 'n02808440', 'n07715103', 'n02481823', 'n04265275', 'n02486410', 'n03447447', 'n04133789', 'n02403003', 'n03444034', 'n02883205', 'n03970156', 'n04465501', 'n02769748', 'n03544143', 'n03733131', 'n01443537', 'n02099712', 'n03100240', 'n02279972', 'n03804744', 'n07875152', 'n07734744', 'n02948072', 'n02099601', 'n02480495', 'n03599486', 'n02074367', 'n03937543', 'n02423022', 'n07920052', 'n03770439', 'n03706229', 'n07615774', 'n02233338', 'n02236044', 'n07871810', 'n04067472', 'n01917289', 'n02795169', 'n02730930', 'n02999410', 'n02085620', 'n03814639', 'n04501370', 'n04398044', 'n03388043', 'n03617480', 'n01774384', 'n04560804', 'n07583066', 'n03179701', 'n04070727', 'n02415577', 'n02841315', 'n03902125', 'n04328186', 'n01945685', 'n07711569', 'n04285008', 'n02113799', 'n03837869', 'n02793495', 'n04118538', 'n04540053', 'n03250847', 'n03838899', 'n01984695', 'n03404251',
                        'n07695742', 'n03089624', 'n02058221', 'n04371430', 'n03424325', 'n04146614', 'n04259630', 'n02123045', 'n04486054', 'n03854065', 'n02094433', 'n04311004', 'n03085013', 'n04532106', 'n03126707', 'n04596742', 'n02823428', 'n02190166', 'n01644900', 'n02410509', 'n02909870', 'n03976657', 'n04179913', 'n04456115', 'n02165456', 'n04008634', 'n02125311', 'n02056570', 'n02814860', 'n02206856', 'n04532670', 'n02906734', 'n12267677', 'n02395406', 'n04487081', 'n03393912', 'n02231487', 'n04399382', 'n01882714', 'n02927161', 'n04023962', 'n03763968', 'n02837789', 'n03983396', 'n01698640', 'n02002724', 'n02977058', 'n03670208', 'n01774750', 'n02950826', 'n04251144', 'n04597913', 'n02226429', 'n02917067', 'n03637318', 'n04356056', 'n03891332', 'n01641577', 'n02106662', 'n01855672', 'n02321529', 'n03992509', 'n02123394', 'n03255030', 'n02791270', 'n03930313', 'n02509815', 'n04099969', 'n01768244', 'n02814533', 'n03014705', 'n02437312', 'n02129165', 'n03355925', 'n07873807', 'n01910747', 'n02843684', 'n02815834', 'n02364673', 'n02699494', 'n04562935', 'n04149813', 'n02132136', 'n07614500', 'n03026506', 'n03980874', 'n04275548', 'n01742172', 'n02504458', 'n09246464', 'n07720875', 'n04376876', 'n07768694', 'n02669723', 'n07579787', 'n09332890', 'n04074963', 'n03977966', 'n02281406', 'n01944390']

        trn_dataset = load_data(data_dir, "train", class_order)
        val_dataset = load_data(data_dir, "val", class_order)
        class_order = list(range(len(class_order)))

        self.novel_classes_per_task = int(self.total_classes * fraction_novel // num_tasks)
        task_to_classes = np.array_split(class_order, num_tasks)

        task_to_novel_class = [t[-self.novel_classes_per_task:] for t in task_to_classes]
        task_to_known_class = [t[:-self.novel_classes_per_task] for t in task_to_classes]
        ssl_transforms = ContrastiveLearningViewGenerator(self.ssl_transform, 2)

        self.train_dataset_dict = self._prepare_dataset(
            trn_dataset, task_to_novel_class, task_to_known_class, self.sl_transform, ssl_transforms, fraction_labeled, fraction_bg)

        self.val_dataset_dict = self._prepare_dataset(
            val_dataset, task_to_novel_class, task_to_known_class, self.test_transform, ssl_transforms, fraction_labeled, fraction_bg)


class TinyImageNetx224(TinyImageNet):
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