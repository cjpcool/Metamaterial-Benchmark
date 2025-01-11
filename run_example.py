from wrappers import MaceVeModel
import os
import torch

DEVICE_ID = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID
device = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')
# device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

model = MaceVeModel(model_name='mace_ve', dataset_name='LatticeModulus',
                    device = device, root_path='./')
model.load_data()
# model.load_model(checkpoint_path='/home/jianpengc/projects/metabench-proj/Metamaterial-Benchmark/checkpoints/mace_ve/poisson_lr_1e-5/32-1000-0.597')
model.load_model()
model.train()
model.evaluate()
model.test()



