"""
Adapted from:
https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/sample.py
"""
import os
from dataclasses import dataclass

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
    batch_size: int = 4

def decode_until_double_newline(token_ids):
    # Convert tokens to string
    full_text = decode(token_ids)
    # Find the first occurrence of double newline
    try:
        end_idx = full_text.index('\n\n') + 2  # +2 to include both newlines
        return full_text[:end_idx]
    except ValueError:
        # If no double newline found, return the full text
        return full_text

if __name__ == "__main__":
    C = parse_config(SampleDefaults)

    print("Using configuration:")
    print(OmegaConf.to_yaml(C))

    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in C.device else "cpu"  # for later use in torch.autocast
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    print("Creating tokenizer...")
    tokenizer = CIFTokenizer()
    print("Tokenizer created")
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
    batch_size = C.batch_size
    # encode the beginning of the prompt
    prompt = C.start
    if prompt.startswith("FILE"):
        # Load prompt from file
        prompt_file = "/home/zhang/Metamaterial-Benchmark/CrystaLLM/test_structures.pkl.gz"  # Remove "FILE:" prefix and whitespace
        with gzip.open(prompt_file, "rb") as f:
            prompts = pickle.load(f)

        # Store all generated results
        all_generated = []
        
        # Process prompts in batches
        for batch_start in range(0, len(prompts), batch_size):
            # Handle last batch which might be smaller
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            
            # Encode all prompts in batch
            batch_ids = []
            for prompt in batch_prompts:
                start_ids = encode(tokenizer.tokenize_cif(prompt))
                batch_ids.append(start_ids)
            
            # Pad sequences to same length
            max_len = max(len(ids) for ids in batch_ids)
            pad_token = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
            padded_ids = [ids + [pad_token]*(max_len - len(ids)) for ids in batch_ids]
            
            # Convert to tensor
            x = torch.tensor(padded_ids, dtype=torch.long, device=C.device)
            
            # Generate for batch
            with torch.no_grad():
                with ctx:
                    y = model.generate(
                        x, 
                        C.max_new_tokens, 
                        temperature=C.temperature, 
                        top_k=C.top_k
                    )
                    
                    # Store generated samples
                    for generated_ids in y:
                        generated = decode_until_double_newline(generated_ids.tolist())
                        all_generated.append(generated)
        
        # Write all results
        for sample_idx, generated in enumerate(all_generated):
            if C.target == "console":
                print(f"Sample {sample_idx}:")
                print(generated)
                print('---------------')
            elif C.target == "file":
                out_dir = os.path.join(C.out_dir, "cond_gen_lattices")
                os.makedirs(out_dir, exist_ok=True)
                fname = os.path.join(out_dir, f"sample_{sample_idx}.cif")
                print(f"writing generated content to {fname} ...")
                with open(fname, "wt") as f:
                    f.write(generated)
    else:
        # Store all generated results
        all_generated = []
        
        # Process multiple samples in smaller batches
        start_ids = encode(tokenizer.tokenize_cif(prompt))
        
        
        # Create progress bar for batches
        num_batches = (C.num_samples + batch_size - 1) // batch_size
        pbar = tqdm(total=num_batches, desc="Generating samples")
        
        for batch_start in range(0, C.num_samples, batch_size):
            # Handle last batch which might be smaller
            batch_size_actual = min(batch_size, C.num_samples - batch_start)
            
            # Create batch of identical prompts
            x = torch.tensor([start_ids] * batch_size_actual, dtype=torch.long, device=C.device)
            
            # Generate for current batch
            with torch.no_grad():
                with ctx:
                    y = model.generate(
                        x, 
                        C.max_new_tokens, 
                        temperature=C.temperature, 
                        top_k=C.top_k
                    )
                    
                    # Store generated samples
                    for generated_ids in y:
                        generated = decode_until_double_newline(generated_ids.tolist())
                        print(generated)
                        all_generated.append(generated)
                        pbar.update(1)
        
        pbar.close()
        
        # Write all results
        for sample_idx, generated in enumerate(all_generated):
            print("A new lattice here: ", generated)
            if C.target == "console":
                print(f"Sample {sample_idx}:")
                print(generated)
                print('---------------')
            elif C.target == "file":
                out_dir = os.path.join(C.out_dir, "uncond_gen_lattices")
                os.makedirs(out_dir, exist_ok=True)
                fname = os.path.join(out_dir, f"sample_{sample_idx}.cif")
                print(f"writing generated content to {fname} ...")
                with open(fname, "wt") as f:
                    f.write(generated)
