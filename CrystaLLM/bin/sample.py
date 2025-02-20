"""
Adapted from:
https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/sample.py
"""
import os
from dataclasses import dataclass
import time  # Add this import at the top with other imports

from contextlib import nullcontext
from omegaconf import OmegaConf
import torch
import gzip
import pickle
from tqdm import tqdm
import random
import numpy as np

from crystallm import (
    parse_config,
    CIFTokenizer,
    GPT,
    GPTConfig,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@dataclass
class SampleDefaults:
    model_dir: str = ""
    out_dir: str = ""  # the path to the directory containing the trained model
    start: str = "\n"  # the prompt; can also specify a file, use as: "FILE:prompt.txt"
    num_samples: int = 2  # number of samples to draw
    max_new_tokens: int = 3000  # number of tokens generated in each sample
    temperature: float = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k: int = 10  # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed: int = 1337
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype: str = "bfloat16"  # 'float32' or 'bfloat16' or 'float16'
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster
    target: str = "console"  # where the generated content will be sent; can also be 'file'

def create_cif_content(lattice):
    """Convert a lattice dictionary into a CIF file prefix."""

    
    y = lattice['mech_props'][0]
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

if __name__ == "__main__":
    C = parse_config(SampleDefaults)

    print("Using configuration:")
    print(OmegaConf.to_yaml(C))

    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in C.device else "cpu"  # for later use in torch.autocast
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    tokenizer = CIFTokenizer()
    encode = tokenizer.encode
    decode = tokenizer.decode

    ckpt_path = os.path.join(C.model_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=C.device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(C.device)
    if C.compile:
        model = torch.compile(model)  # requires PyTorch 2.0 (optional)
    cond = False    
    # encode the beginning of the prompt
    prompt = C.start
    if prompt.startswith("FILE:"):
        print("Using test set for conditional generation")
        cond = True
        x = []
        test_dataset = pickle.load(open("/home/zhang/Metamaterial-Benchmark/unit_cell_catalog/test_set.pkl", "rb"))
        for lattice in test_dataset:
            cif_content = create_cif_content(lattice)
            start_ids = encode(tokenizer.tokenize_cif(cif_content))
            x.append(torch.tensor(start_ids, dtype=torch.long, device=C.device)[None, ...])
        num_samples = len(x)
    else:
        print("Using prompt for unconditional generation")
        start_ids = encode(tokenizer.tokenize_cif(prompt))
        x = torch.tensor(start_ids, dtype=torch.long, device=C.device)[None, ...]
        num_samples = C.num_samples
    # run generation
    with torch.no_grad():
        with ctx:
            generation_times = []  # List to store generation times
            for k in range(num_samples):
                start_time = time.time()  # Start timing
                if cond:
                    y = model.generate(x[k], C.max_new_tokens, temperature=C.temperature, top_k=C.top_k)
                else:
                    y = model.generate(x, C.max_new_tokens, temperature=C.temperature, top_k=C.top_k)
                end_time = time.time()  # End timing
                generation_time = end_time - start_time  # Calculate duration
                generation_times.append(generation_time)

                generated = decode(y[0].tolist())

                if C.target == "console":
                    print(generated)
                    print(f'Generation time: {generation_time:.2f} seconds')
                    print('---------------')
                elif C.target == "file":
                    print(generated)
                    print(f'Generation time: {generation_time:.2f} seconds')
                    print('---------------')
                    if not cond:
                        out_dir = os.path.join(C.out_dir, "uncond_gen_lattices")
                    else:
                        out_dir = os.path.join(C.out_dir, "cond_gen_lattices")
                    os.makedirs(out_dir, exist_ok=True)
                    fname = os.path.join(out_dir, f"sample_{k}.cif")
                    print(f"writing generated content to {fname} ...")
                    with open(fname, "wt") as f:
                        f.write(generated)
            
            # Calculate and print average generation time
            avg_generation_time = sum(generation_times) / len(generation_times)
            print(f'\nAverage generation time: {avg_generation_time:.2f} seconds')

