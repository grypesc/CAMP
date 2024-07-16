import os
import numpy as np

from PIL import Image
from torchvision import transforms

from datasets.base import BaseDataModule, MemoryDataset, ContrastiveLearningViewGenerator


class CUB200(BaseDataModule):
    image_size = (32, 32)
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

    def __init__(self, args, total_classes=200):
        super().__init__(args)
        self.total_classes = total_classes
        self.create_datasets(args.data_dir, args.num_tasks, args.fraction_novel, args.fraction_labeled, args.fraction_bg)
        self.classes_per_task = self.total_classes // args.num_tasks

    def _load_data_(self, data_dir, classs_order):
        id_to_classes = {i: classs_order[i] for i in range(len(classs_order))}
        train_images, train_labels = [], []
        val_images, val_labels = [], []

        path = os.path.join(data_dir, "CUB_200_2011", "images")
        for i, class_name in id_to_classes.items():
            class_path = os.path.join(path, class_name)
            image_names = os.listdir(class_path)
            for j, image_name in enumerate(image_names):
                image = Image.open(os.path.join(class_path, image_name)).convert('RGB').resize(self.image_size)
                # Put ~10% of class images in val
                if j > 0.1 * len(image_names):
                    train_images.append(np.array(image))
                    train_labels.append(i)
                else:
                    val_images.append(np.array(image))
                    val_labels.append(i)

        train_images = np.stack(train_images)
        train_labels = np.stack(train_labels)
        val_images = np.stack(val_images)
        val_labels = np.stack(val_labels)
        return MemoryDataset(train_images, train_labels, None), MemoryDataset(val_images, val_labels, None)

    def create_datasets(self, data_dir, num_tasks, fraction_novel=0.2, fraction_labeled=0.2, fraction_bg=0.05):
        """Produces train and val datasets. Tiny imagenet has different data structure for train and val, so it is a bit messy"""
        class_order = ['040.Olive_sided_Flycatcher', '094.White_breasted_Nuthatch', '035.Purple_Finch', '193.Bewick_Wren', '176.Prairie_Warbler', '011.Rusty_Blackbird', '106.Horned_Puffin', '137.Cliff_Swallow', '083.White_breasted_Kingfisher', '014.Indigo_Bunting', '060.Glaucous_winged_Gull', '081.Pied_Kingfisher', '038.Great_Crested_Flycatcher', '121.Grasshopper_Sparrow', '156.White_eyed_Vireo', '028.Brown_Creeper', '123.Henslow_Sparrow', '010.Red_winged_Blackbird', '088.Western_Meadowlark', '025.Pelagic_Cormorant', '082.Ringed_Kingfisher', '155.Warbling_Vireo', '183.Northern_Waterthrush', '170.Mourning_Warbler', '144.Common_Tern', '136.Barn_Swallow', '173.Orange_crowned_Warbler', '046.Gadwall', '013.Bobolink', '192.Downy_Woodpecker', '098.Scott_Oriole', '184.Louisiana_Waterthrush', '142.Black_Tern', '161.Blue_winged_Warbler', '147.Least_Tern', '133.White_throated_Sparrow', '196.House_Wren', '114.Black_throated_Sparrow', '180.Wilson_Warbler', '116.Chipping_Sparrow', '034.Gray_crowned_Rosy_Finch', '171.Myrtle_Warbler', '052.Pied_billed_Grebe', '120.Fox_Sparrow', '127.Savannah_Sparrow', '057.Rose_breasted_Grosbeak', '029.American_Crow', '071.Long_tailed_Jaeger', '027.Shiny_Cowbird', '179.Tennessee_Warbler', '036.Northern_Flicker', '022.Chuck_will_Widow', '089.Hooded_Merganser', '065.Slaty_backed_Gull', '151.Black_capped_Vireo', '079.Belted_Kingfisher', '073.Blue_Jay', '154.Red_eyed_Vireo', '048.European_Goldfinch', '113.Baird_Sparrow', '077.Tropical_Kingbird', '047.American_Goldfinch', '039.Least_Flycatcher', '178.Swainson_Warbler', '126.Nelson_Sharp_tailed_Sparrow', '009.Brewer_Blackbird', '148.Green_tailed_Towhee', '153.Philadelphia_Vireo', '199.Winter_Wren', '165.Chestnut_sided_Warbler', '138.Tree_Swallow', '167.Hooded_Warbler', '198.Rock_Wren', '095.Baltimore_Oriole', '037.Acadian_Flycatcher', '099.Ovenbird',
                       '075.Green_Jay', '119.Field_Sparrow', '043.Yellow_bellied_Flycatcher', '086.Pacific_Loon', '024.Red_faced_Cormorant', '008.Rhinoceros_Auklet', '042.Vermilion_Flycatcher', '169.Magnolia_Warbler', '050.Eared_Grebe', '107.Common_Raven', '063.Ivory_Gull', '168.Kentucky_Warbler', '091.Mockingbird', '074.Florida_Jay', '117.Clay_colored_Sparrow', '004.Groove_billed_Ani', '195.Carolina_Wren', '150.Sage_Thrasher', '097.Orchard_Oriole', '080.Green_Kingfisher', '085.Horned_Lark', '066.Western_Gull', '162.Canada_Warbler', '145.Elegant_Tern', '053.Western_Grebe', '093.Clark_Nutcracker', '122.Harris_Sparrow', '026.Bronzed_Cowbird', '174.Palm_Warbler', '129.Song_Sparrow', '146.Forsters_Tern', '092.Nighthawk', '005.Crested_Auklet', '163.Cape_May_Warbler', '131.Vesper_Sparrow', '128.Seaside_Sparrow', '182.Yellow_Warbler', '018.Spotted_Catbird', '186.Cedar_Waxwing', '159.Black_and_white_Warbler', '200.Common_Yellowthroat', '194.Cactus_Wren', '044.Frigatebird', '101.White_Pelican', '054.Blue_Grosbeak', '177.Prothonotary_Warbler', '002.Laysan_Albatross', '191.Red_headed_Woodpecker', '007.Parakeet_Auklet', '064.Ring_billed_Gull', '115.Brewer_Sparrow', '118.House_Sparrow', '157.Yellow_throated_Vireo', '017.Cardinal', '058.Pigeon_Guillemot', '003.Sooty_Albatross', '072.Pomarine_Jaeger', '102.Western_Wood_Pewee', '049.Boat_tailed_Grackle', '016.Painted_Bunting', '110.Geococcyx', '149.Brown_Thrasher', '105.Whip_poor_Will', '158.Bay_breasted_Warbler', '143.Caspian_Tern', '051.Horned_Grebe', '109.American_Redstart', '059.California_Gull', '001.Black_footed_Albatross', '068.Ruby_throated_Hummingbird', '164.Cerulean_Warbler', '185.Bohemian_Waxwing', '061.Heermann_Gull', '135.Bank_Swallow', '124.Le_Conte_Sparrow', '189.Red_bellied_Woodpecker', '033.Yellow_billed_Cuckoo', '152.Blue_headed_Vireo', '062.Herring_Gull', '134.Cape_Glossy_Starling', '006.Least_Auklet', '166.Golden_winged_Warbler', '041.Scissor_tailed_Flycatcher', '096.Hooded_Oriole', '112.Great_Grey_Shrike', '190.Red_cockaded_Woodpecker', '087.Mallard', '100.Brown_Pelican', '188.Pileated_Woodpecker', '181.Worm_eating_Warbler', '069.Rufous_Hummingbird', '030.Fish_Crow', '125.Lincoln_Sparrow', '015.Lazuli_Bunting', '139.Scarlet_Tanager', '045.Northern_Fulmar', '056.Pine_Grosbeak', '031.Black_billed_Cuckoo', '141.Artic_Tern', '090.Red_breasted_Merganser', '104.American_Pipit', '187.American_Three_toed_Woodpecker', '032.Mangrove_Cuckoo', '078.Gray_Kingbird', '160.Black_throated_Blue_Warbler', '020.Yellow_breasted_Chat', '021.Eastern_Towhee', '172.Nashville_Warbler', '055.Evening_Grosbeak', '103.Sayornis', '023.Brandt_Cormorant', '197.Marsh_Wren', '012.Yellow_headed_Blackbird', '067.Anna_Hummingbird', '130.Tree_Sparrow', '084.Red_legged_Kittiwake', '070.Green_Violetear', '175.Pine_Warbler', '108.White_necked_Raven', '111.Loggerhead_Shrike', '076.Dark_eyed_Junco', '019.Gray_Catbird',
                       '140.Summer_Tanager', '132.White_crowned_Sparrow'][:self.total_classes]

        trn_dataset, val_dataset = self._load_data_(data_dir, class_order)
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


class CUB200x224(CUB200):
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

    def __init__(self, args, total_classes=200):
        super().__init__(args, total_classes)


class CUB200Subset(CUB200):
    def __init__(self, args):
        super().__init__(args, total_classes=10)


class CUB200x224Subset(CUB200x224):
    def __init__(self, args):
        super().__init__(args, total_classes=10)
