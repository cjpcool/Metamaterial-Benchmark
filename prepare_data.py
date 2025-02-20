import random
import os
import pickle
from datasets_scripts.dataset_truss import LatticeModulus
import numpy as np
import torch
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def filter_data(data):
    """
    Filters the data object to retain only the specified attributes.

    Args:
        data: A data object with multiple attributes.

    Returns:
        A dictionary with only the selected attributes.
    """
    return {
        "edge_index": np.array(data.edge_index),
        "mech_props": np.array(data.y),
        "frac_coords": np.array(data.frac_coords),
        "lengths": np.array(data.lengths),
        "angles": np.array(data.angles),
    }

def split_dataset(dataset, train_size=8000, val_size=2000, shuffle=True):
    """
    Splits the dataset into train, validation, and test sets with fixed sizes.

    Args:
        dataset: List of data objects.
        train_size: Fixed size of training set.
        val_size: Fixed size of validation set.
        shuffle: Whether to shuffle the data before splitting.

    Returns:
        train_set, val_set, test_set: Split datasets.
    """
    if len(dataset) < (train_size + val_size):
        raise ValueError("Dataset is too small for the requested split sizes")

    if shuffle:
        random.shuffle(dataset)

    train_set = dataset[:train_size]
    val_set = dataset[train_size:train_size + val_size]
    test_set = dataset[train_size + val_size:train_size + val_size + 2000]

    return train_set, val_set, test_set


def save_dataset(dataset, file_path):
    """
    Saves the filtered dataset to a file.

    Args:
        dataset: List of filtered data objects to save.
        file_path: File path to save the dataset.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    # Load your dataset (assuming it is iterable like a list)
    lattice = LatticeModulus(data_path="LatticeModulus")

    # Convert to a list for easier processing (if not already a list)
    dataset = [filter_data(data) for data in lattice]

    # Split the dataset
    train_set, val_set, test_set = split_dataset(dataset)

    # Create output directory if it does not exist
    output_dir = "unit_cell_catalog"
    os.makedirs(output_dir, exist_ok=True)

    print("train_set: ", len(train_set))
    print("val_set: ", len(val_set))
    print("test_set: ", len(test_set))

    pickle.dump(train_set, open("unit_cell_catalog/train_set.pkl", "wb"))
    pickle.dump(val_set, open("unit_cell_catalog/val_set.pkl", "wb"))
    pickle.dump(test_set, open("unit_cell_catalog/test_set.pkl", "wb"))