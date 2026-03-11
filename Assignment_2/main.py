import random

import numpy as np
import torch

from Assignment_2.ablation import print_ablation_results
from Assignment_2.data_processing import test_loader, train_loader, val_loader
from Assignment_2.error_analysis import print_misclassified_examples
from Assignment_2.evaluation import (
    print_conf_mat,
    print_evaluation_results,
    print_learning_curves,
)


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

# Processing the data and creating the train, validation, and test loaders
train_set = train_loader
val_set = val_loader
test_set = test_loader

# Training and evaluating the models, and then printing the results
print("Evaluating the models...")
print_evaluation_results()

print("Plotting confusion matrices...")
print_conf_mat()

print("Plotting learning curves...")
print_learning_curves()

# Running the ablation study and printing the results
print("Running ablation study on dropout...")
print_ablation_results()

# Getting and printing the missclassified examples from the test set
print("Missclassified examples from the test set:")
print_misclassified_examples()
