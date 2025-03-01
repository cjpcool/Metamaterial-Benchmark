import wandb

from wrappers import MaceVeModel
import os
import torch
import pandas as pd
import random

DEVICE_ID = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID
device = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')
# device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')




results = []
# props = ['shear','young',  'poisson']
save_name = 'results_maceve_props_2.csv'
prop = 'young'
i=3

model = MaceVeModel(model_name='mace_ve', dataset_name='LatticeModulus',
                    device=device, root_path='./')
model.config['wandb_args']['use_wandb']=False
model.config['training']['pred_property'] = prop
model.config['training']['save_dir'] = f'./checkpoints/mace_ve/{prop}_{i}'
model.config['training']['log_dir'] = f'./logs/mace_ve/{prop}_{i}'
model.config['wandb_args']['save_name'] = f'mace_ve-{prop}-{i}'
# model.config['data_seed'] = random.randint(1, 10000)
model.load_data()

model.load_model(checkpoint_path=f'./checkpoints/mace_ve/{prop}_{i}/best_model.pth')


r2, nrmse, mae = model.test()