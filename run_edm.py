import wandb

from wrappers import EDMModel
import os
import torch
import pandas as pd
import random

DEVICE_ID = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID
device = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')
# device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

model = EDMModel(model_name='edm', dataset_name='LatticeModulus',
                            device=device, root_path='./')


model.load_data()
model.load_model()
# model.train()
model.generate(generators_path='./checkpoints/edm/debug_21',  n_sweeps=500,n_frames=2, save_path='gen_results/edm/save_results', use_test_data=True)
