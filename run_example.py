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
props = ['young']
for prop in props:
        i=3
        print(f'Run {prop} {i}th time.')
        model = MaceVeModel(model_name='mace_ve', dataset_name='LatticeModulus',
                            device=device, root_path='./')
        model.config['training']['pred_property'] = prop
        model.config['training']['save_dir'] = f'./checkpoints/mace_ve/{prop}_{i}'
        model.config['training']['log_dir'] = f'./logs/mace_ve/{prop}_{i}'
        # model.config['wandb_args']['use_wandb'] = False
        model.config['wandb_args']['save_name'] = f'mace_ve-{prop}-{i}'
        # model.config['data_seed'] = random.randint(1, 10000)
        model.load_data()
        if prop == 'shear':
            model.load_model(checkpoint_path=f'./checkpoints/mace_ve/{prop}_{i}/best_model.pth')
        else:
            model.load_model()
        # model.load_model()
        # model.train()
        # model.evaluate()
        # model.load_model(checkpoint_path='/home/jianpengc/projects/metabench-proj/Metamaterial-Benchmark/checkpoints/mace_ve/young_0/best_model.pth')
        model.load_model(checkpoint_path=f'./checkpoints/mace_ve/{prop}_{i}/best_model.pth')
        r2, nrmse, mae = model.test()
        wandb.finish()
        results.append({
            'prop': prop,
            'R2': r2,
            'NRMSE': nrmse,
            'MAE': mae
        })
        df = pd.DataFrame(results)
        if os.path.exists(save_name):
            df1 = pd.read_csv(save_name)
            pd.concat([df1, df]).to_csv(save_name)
        else:
            df.to_csv(save_name,index=False)