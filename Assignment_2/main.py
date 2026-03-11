import random

import numpy as np
import torch


def set_seed(seed: int = 13) -> None:
    """
    Sets random seeds for reproducibility.

    Args:
        seed (int): The seed value to use for random number generators. Defaults to 13

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(13)

# Checking the available GPU and choosing a device
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")
