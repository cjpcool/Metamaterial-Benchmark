from wrappers import MaceVeModel
import os
import torch

DEVICE_ID = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID

model = MaceVeModel(model_name='mace_ve', dataset_name='LatticeModulus',
                    device = torch.device("cuda"), root_path='./')
model.load_data()
model.load_model()
model.train()




