from wrappers import MaceVeModel
import os
import torch

DEVICE_ID = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID
device = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')
# device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

model = MaceVeModel(model_name='mace_ve', dataset_name='LatticeModulus',
                    device = device, root_path='./')
model.load_data()
model.train()
model.evaluate()
# model.load_model(checkpoint_path='/home/jianpengc/projects/metabench-proj/Metamaterial-Benchmark/checkpoints/mace_ve/uni_density/117-4800-0.003')
model.load_model(checkpoint_path='./checkpoints/mace_ve/uni_density/best_model.pth')
model.test()



