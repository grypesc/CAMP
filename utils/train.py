import logging
import os
import random
import time
import numpy as np
import torch


def dict_to_device(tensor_dict, device):
    for k, v in tensor_dict.items():
        tensor_dict[k] = v.to(device, non_blocking=True)
    return tensor_dict


def create_log_dir(base_log_dir):
    current_time = time.strftime('%Y-%m-%d_%H:%M:%S')
    checkpoint_dir = os.path.join(base_log_dir, current_time)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def seed_everything(seed: int):
    logging.info(f"Global seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
