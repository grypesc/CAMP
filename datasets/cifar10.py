from torchvision.datasets import CIFAR10 as CIFAR10TV

from datasets.base_cifar import BaseCIFAR, base_cifar_224_wrapper


class CIFAR10(BaseCIFAR):
    original_class_order = [5, 3, 4, 9, 1, 0, 2, 7, 6, 8]
    
    def __init__(self, args):
        super().__init__(args, 10)

    def get_dataset_class(self):
        return CIFAR10TV


CIFAR10x224 = base_cifar_224_wrapper(CIFAR10)
