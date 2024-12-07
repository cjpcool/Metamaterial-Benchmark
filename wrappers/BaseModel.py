import os
from abc import ABC, abstractmethod
import yaml
import matplotlib.pyplot as plt


class BaseModel(ABC):
    def __init__(self, model_name, dataset_name, root_path='../'):
        config_path = os.path.join(root_path, 'configs', model_name, dataset_name+'_config.yml')
        self.config = self.load_config(config_path)
        self.model = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.metrics = {}


    def load_data(self):
        pass


    def evaluate(self):
        pass

    def test(self):
        pass

    def visualize(self, path='results/plots/'):
        pass

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
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
