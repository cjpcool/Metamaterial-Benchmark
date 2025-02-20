import torch
import os
from typing import List, Callable

from bokeh.core.has_props import abstract
from sklearn.utils import shuffle
from torch_geometric.data import (
    InMemoryDataset,
    download_url,
)
from torch.utils.data import Dataset
from abc import abstractmethod


class LatticeTruss(InMemoryDataset):
    def __init__(self, data_path: str,
                 file_name='data'):
        self.data_path = data_path
        self.file_name = file_name

        super().__init__(root=data_path)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_data_exist(self):
        return os.path.exists(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.file_name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.file_name, 'processed')

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    @property
    def raw_file_names(self) -> str:
        name = self.file_names[self.name]
        return name

    def download(self):
        url = self.downloadurl
        path = download_url(url, self.raw_dir)
        return path

    @abstractmethod
    def process(self):
        pass

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.LongTensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict



class LatticeImage(Dataset):
    # TODO
    def __init__(self, data_path: str,
                 file_name='data'):
        return NotImplementedError

class LatticeCloudPoint(Dataset):
    # TODO
    def __init__(self, data_path: str,
                 file_name='data'):

        return NotImplementedError