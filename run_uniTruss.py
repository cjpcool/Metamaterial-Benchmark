import wandb

from wrappers import UniTrussModel
import os
import torch
import pandas as pd
import random

DEVICE_ID = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID
device = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')
# device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')




results = []
# props = ['young','shear','poisson']
props = ['young']
save_name = 'results_unitruss_props_2.csv'
for prop in props:
        i = 3
        print(f'Run {prop} {i}th time.')
        model = UniTrussModel(model_name='uni_truss', dataset_name='LatticeModulus',
                              device=device, root_path='./')
        model.config['pred_property'] = prop
        model.config['output_dir'] = f'./checkpoints/uni_truss/{prop}_{i}'
        # model.config['log_dir'] = f'./logs/mace_ve/{prop}_{i}'
        model.config['wandb_args']['save_name'] = f'uni_truss-{prop}-{i}'
        # model.config['data_seed'] = random.randint(1, 10000)
        model.load_data()
        model.load_model()
        # model.train()
        # model.evaluate()
        # model.load_model(checkpoint_path='/home/jianpengc/projects/metabench-proj/Metamaterial-Benchmark/checkpoints/mace_ve/uni_density/117-4800-0.003')
        model.load_model(checkpoint_path=model.config['output_dir'])
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
            df.to_csv(save_name, index=False)