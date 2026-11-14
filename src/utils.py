import random

import numpy as np
import torch


def get_device():
    """Выбирает, на чём считать: видеокарта NVIDIA, GPU Mac или процессор."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    """Фиксирует случайность, чтобы эксперименты повторялись один в один."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
