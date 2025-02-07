from argparse import Namespace
import datetime

import wandb
from scipy.stats.tests.test_continuous_fit_censored import optimizer
from tqdm import tqdm

from models.mace_ve.lattices.utils import elasticity_func
from wrappers.BaseModel import BaseModel
import torch
from models.mace_ve import EnergyEquivGNN
import time
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
from torch.nn import functional as F
import numpy as np
from evaluation.prediction_eval import calculate_metrics








class MaceVeModel(BaseModel):

    def __init__(self, model_name, dataset_name, device=torch.device('cuda'), root_path='../'):
        super(MaceVeModel, self).__init__(model_name, dataset_name, device, root_path)
        self.ckpt = None