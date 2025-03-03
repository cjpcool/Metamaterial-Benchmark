from wrappers.CrystaLLM import CrystaLLM
import torch

DEVICE_ID = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID
device = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')


model = CrystaLLM(model_name='crystallm', dataset_name='LatticeModulus',
                            device=device, root_path='./')


model.load_model()

# Unconditional generation
# model.generate()

# Condional generation
properties = torch.randn(12)
model.generate(conditions=properties)




