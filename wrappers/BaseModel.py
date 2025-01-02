import os
from abc import ABC, abstractmethod
import torch
import yaml
import matplotlib.pyplot as plt
from datasets import LatticeModulus, LatticeStiffness


class BaseModel(ABC):
    def __init__(self, model_name, dataset_name, device=torch.device('cuda'), root_path='../'):
        config_path = os.path.join(root_path, 'configs', model_name, dataset_name+'_config.yml')
        self.config = self.load_config(config_path)
        self.model = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.device=device
        self.metrics = {}


    def load_data(self):
        dataset = eval("{}('{}')".format(self.config['data_name'], self.config['data_path'], encoding='utf-8'))

        split_idx = dataset.get_idx_split(len(dataset), train_size=self.config['train_size'],
                                          valid_size=self.config['valid_size'], seed=self.config['data_seed'])
        print(split_idx.keys())
        print(dataset[split_idx['train']])
        self.train_data, self.val_data, self.test_data = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[
            split_idx['test']]

        return self.train_data, self.val_data, self.test_data


    def evaluate(self):
        pass

    def test(self):
        pass

    def visualize(self, path='results/plots/'):
        pass

    def load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def save_results(self, path='results/'):
        pass


    @abstractmethod
    def train(self):
        pass
