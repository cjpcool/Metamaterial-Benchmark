import torch
from wrappers.BaseModel import BaseModel
from dataclasses import dataclass
import os
from typing import List, Dict, Tuple
import numpy as np

@dataclass
class SampleDefaults:
    model_dir: str = ""
    out_dir: str = ""  # the path to the directory containing the trained model
    start: str = "\n"  # the prompt; can also specify a file, use as: "FILE:prompt.txt"
    num_samples: int = 2  # number of samples to draw
    max_new_tokens: int = 2048  # number of tokens generated in each sample
    temperature: float = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k: int = 10  # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed: int = 1337
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype: str = "bfloat16"  # 'float32' or 'bfloat16' or 'float16'
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster
    target: str = "console"

from models.crystallm.CrystaLLM import (
    parse_config,
    CIFTokenizer,
    GPT,
    GPTConfig,
)

class CrystaLLM(BaseModel):

    def __init__(self, model_name, dataset_name, device=torch.device('cuda'), root_path='../'):
        super(CrystaLLM, self).__init__(model_name, dataset_name, device, root_path)
        self.dtype = torch.float32
        self.config = parse_config(SampleDefaults)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True 

    def load_data(self):
        pass

    def load_model(self, checkpoint="models/crystallm/metamat_crystallm_v1_large"):
        
        # Load the tokenizer
        self.tokenizer = CIFTokenizer()
        self.encode = self.tokenizer.encode
        self.decode = self.tokenizer.decode

        # Load the model checkpoint
        ckpt_path = os.path.join(checkpoint, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        if self.config.compile:
            model = torch.compile(model)
        self.model = model


    def generate(self, path='results/crystallm', conditions=None):
        '''
        conditions: shape (12, ), Young's Modulus, Shear Modulus, Poisson's Ratio
        '''
        if conditions is None:
            prompt = "lattice_data_"
        else:
            prompt = create_cif_prefix(conditions)
        start_ids = self.encode(self.tokenizer.tokenize_cif(prompt))
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]
        generated = self.model.generate(x, self.config.max_new_tokens, temperature=self.config.temperature, top_k=self.config.top_k)
        generated_text = self.decode(generated[0].tolist())
        parse_cif_file(generated_text, path)


def create_cif_prefix(y):
    """Convert a lattice dictionary into a CIF file prefix."""

    
    youngs_modulus = y[0:3]
    shear_modulus = y[3:6]
    poisson_ratio = y[6:12]
    
    cif_content = f"""lattice_data_
_mechanical
_youngs_modulus
_x     {youngs_modulus[0]:.6f}
_y     {youngs_modulus[1]:.6f}
_z     {youngs_modulus[2]:.6f}
_shear_modulus
_xy     {shear_modulus[0]:.6f}
_xz     {shear_modulus[1]:.6f}
_yz     {shear_modulus[2]:.6f}
_poisson_ratio
_xy     {poisson_ratio[0]:.6f}
_xz     {poisson_ratio[1]:.6f}
_yx     {poisson_ratio[2]:.6f}
_yz     {poisson_ratio[3]:.6f}
_zx     {poisson_ratio[4]:.6f}
_zy     {poisson_ratio[5]:.6f}"""
    
    return cif_content



def parse_cif_file(cif_text: str, saving_path: str) -> Dict:
    """Parse a single .cif file and return its parameters."""

    lines = cif_text.split('\n')
    structure = {
        'lengths': [],
        'angles': [],
        'coordinates': [],
        'edge_index': [],
        'youngs_modulus': [],
        'shear_modulus': [],
        'poisson_ratio': []
    }
    
    # Parse lattice parameters and mechanical properties
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        
        # Parse mechanical properties
        if line.startswith('_youngs_modulus'):
            current_section = 'youngs'
            continue
        elif line.startswith('_shear_modulus'):
            current_section = 'shear'
            continue
        elif line.startswith('_poisson_ratio'):
            current_section = 'poisson'
            continue
        elif line.startswith('_params'):
            current_section = None
            continue
            
        # Parse values based on their prefixes
        if current_section == 'youngs' and (line.startswith('_x') or line.startswith('_y') or line.startswith('_z')):
            try:
                structure['youngs_modulus'].append(float(parts[-1]))
            except (ValueError, IndexError):
                continue
                
        elif current_section == 'shear' and (line.startswith('_xy') or line.startswith('_xz') or line.startswith('_yz')):
            try:
                structure['shear_modulus'].append(float(parts[-1]))
            except (ValueError, IndexError):
                continue
                
        elif current_section == 'poisson' and (line.startswith('_xy') or line.startswith('_xz') or 
                                             line.startswith('_yx') or line.startswith('_yz') or
                                             line.startswith('_zx') or line.startswith('_zy')):
            try:
                structure['poisson_ratio'].append(float(parts[-1]))
            except (ValueError, IndexError):
                continue
                
        # Parse lattice parameters
        elif line.startswith('_cell_length'):
            try:
                structure['lengths'].append(float(parts[-1]))
            except (ValueError, IndexError):
                continue
        elif line.startswith('_cell_angle'):
            try:
                structure['angles'].append(float(parts[-1]))
            except (ValueError, IndexError):
                continue

    # Parse atomic coordinates
    for line in lines:
        line = line.strip()
        if line.startswith('atom_'):
            try:
                parts = line.split()
                coords = [float(parts[1]), float(parts[2]), float(parts[3])]
                structure['coordinates'].append(coords)
            except (ValueError, IndexError):
                continue

    # Parse edge indices
    edge_section = False
    for line in lines:
        line = line.strip()
        if '_edge_from_atom' in line:
            edge_section = True
            continue
        if edge_section and line and not line.startswith('_'):
            try:
                parts = line.split()
                if len(parts) == 2:
                    from_atom, to_atom = map(int, parts)
                    structure['edge_index'].append([from_atom, to_atom])
            except (ValueError, IndexError):
                continue

    # Convert to numpy arrays
    structure['lengths'] = np.array(structure['lengths'])
    structure['angles'] = np.array(structure['angles'])
    structure['coordinates'] = np.array(structure['coordinates'], dtype=np.float64)
    structure['edge_index'] = np.array(structure['edge_index'])
    structure['youngs_modulus'] = np.array(structure['youngs_modulus'])
    structure['shear_modulus'] = np.array(structure['shear_modulus'])
    structure['poisson_ratio'] = np.array(structure['poisson_ratio'])
    structure['mech_props'] = np.concatenate([
        structure['youngs_modulus'],
        structure['shear_modulus'],
        structure['poisson_ratio']
    ])
    # Debug print
    # print(f"Young's modulus: {len(structure['youngs_modulus'])}")
    # print(f"Shear modulus: {len(structure['shear_modulus'])}")
    # print(f"Poisson ratio: {len(structure['poisson_ratio'])}")
    
    # Verify we have the correct number of parameters
    if len(structure['lengths']) != 3 or len(structure['angles']) != 3:
        raise ValueError("Missing lattice parameters")
    if (len(structure['youngs_modulus']) != 3 or 
        len(structure['shear_modulus']) != 3 or 
        len(structure['poisson_ratio']) != 6):
        raise ValueError("Missing mechanical properties")
    
    saving_path = os.path.join(saving_path, "gen_result")
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(saving_path), exist_ok=True)
    np.savez(saving_path,
            atom_types=None,
            lengths=structure['lengths'],
            angles=structure['angles'],
            frac_coords=structure['coordinates'],
            edge_index=structure['edge_index'],
            prop_list=structure['mech_props'],
            )