import wandb

from wrappers import GeoLDMModel
import os
import torch
import pandas as pd
import random

DEVICE_ID = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID
device = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')
# device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

model = GeoLDMModel(model_name='geoldm', dataset_name='LatticeModulus',
                            device=device, root_path='./')


model.load_data()
model.load_model()
# model.train()
model.generate(generators_path='./checkpoints/geoldm/debug_11', n_sweeps=500,n_frames=2, save_path='gen_results/geoldm/save_results_non_dataset') # n_frames is samplese per number of nodes
