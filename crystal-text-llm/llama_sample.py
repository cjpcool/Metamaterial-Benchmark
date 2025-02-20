"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import random
import argparse
import pandas as pd
import numpy as np
import pickle
import torch
from transformers import (
    LlamaForCausalLM, LlamaTokenizer
)
from peft import PeftModel
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from llama_finetune import (
    parse_mechanical_properties,   
    MAX_LENGTH
)
# from templating import make_swap_table
from tqdm import tqdm
import matplotlib.pyplot as plt

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import time

def parse_fn(gen_str):
    try:
        sections = gen_str.strip().split('\n\n')
        if len(sections) == 3:
            lattice_vectors_str, coords_str, edges_str = sections
        elif len(sections) == 2:
            lattice_vectors_str, coords_str = sections
            edges_str = None
        elif len(sections) == 1:
            lattice_vectors_str = sections[0]
            coords_str = None
            edges_str = None
        else:
            raise ValueError(f"Expected 1-3 sections, got {len(sections)}")
        
        lattice_vectors_lines = lattice_vectors_str.strip().split('\n')
        lengths = np.array([float(x.strip(',')) for x in lattice_vectors_lines[0].split()])
        angles = np.array([float(x.strip(',')) for x in lattice_vectors_lines[1].split()])
        
        if coords_str:
            coords_lines = coords_str.strip().split('\n')
            # Filter coords_lines to only keep lines with 4 items (index + 3 coordinates)
            coords_lines = [line for line in coords_lines if len(line.split()) == 4]
            coords = np.array([list(map(float, line.split()[1:])) for line in coords_lines])
        else:
            coords = None
        
        if edges_str:
            edges_lines = edges_str.strip().split('\n')
            edges_lines = [line for line in edges_lines if len(line.split()) == 2]
            edges = np.array([list(map(int, line.split())) for line in edges_lines])
        else:
            edges = None
        
        return lengths, angles, coords, edges
    
    except Exception as e:
        print(f"Error parsing generated string: {e}")
        print(f"Generated string:\n{gen_str}")
        raise


def visualizeLattice(nodes, struts, save_dir="", dpi=150):
    """
    Visualize the lattice structure from the provided nodes and struts with uniform
    colors for nodes and edges, and uniform edge width, randomly chosen for each plot.
    
    Parameters:
        nodes (np.array): Array of node coordinates.
        struts (np.array): Array of strut connections.
        save_dir (str): Directory to save the plot.
        dpi (int): Dots per inch setting for the plot resolution.
    """
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    node_color = np.random.rand(3)
    edge_color = np.random.rand(3)
    edge_width = random.uniform(0.5, 3.0)
    

    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c=[node_color], s=60, edgecolor='black')
    
    for strut in struts:
        start_node = nodes[strut[0]]
        end_node = nodes[strut[1]]
        ax.plot([start_node[0], end_node[0]], 
                [start_node[1], end_node[1]], 
                [start_node[2], end_node[2]], 
                color=edge_color, linewidth=edge_width)
    
    # Set random camera angle
    elev = random.uniform(0, 90)
    azim = random.uniform(0, 360)
    ax.view_init(elev=elev, azim=azim)
    
    
    # Adjust plot limits to ensure all nodes are visible
    ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
    ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
    ax.set_zlim(nodes[:, 2].min(), nodes[:, 2].max())
    
    if save_dir:
        plt.savefig(save_dir, bbox_inches='tight')
    plt.close()

def prepare_model_and_tokenizer(args):
    llama_options = args.model_name.split("-")
    is_chat = len(llama_options) == 2
    model_size = llama_options[0]

    def llama2_model_string(model_size, chat):
        chat = "chat-" if chat else ""
        return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

    model_string = llama2_model_string(model_size, is_chat)
    
    model = LlamaForCausalLM.from_pretrained(
        model_string,
        load_in_8bit=True,
        device_map="auto",
    )

    tokenizer = LlamaTokenizer.from_pretrained(
        model_string,
        model_max_length=MAX_LENGTH,
        padding_side="right",
        use_fast=False,
    )

    model.eval()

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=tokenizer,
        model=model,
    )

    model = PeftModel.from_pretrained(model, args.model_path, device_map="auto")
    
    return model, tokenizer

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict, 
    llama_tokenizer, 
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = llama_tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(llama_tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def calculate_generation_time(start_time, num_samples):
    """Calculate average time per sample."""
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_samples
    print(f"\nTotal generation time: {total_time:.2f}s")
    print(f"Average time per sample: {avg_time:.2f}s")
    return avg_time


def unconditional_sample(args):
    model, tokenizer = prepare_model_and_tokenizer(args)
    print("Model and tokenizer prepared")
    prompts = []
    for _ in range(args.num_samples):
        prompt = "Below is a description of a truss metamaterial. "
        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "then the coordinates for each node within the lattice, and the edge connections between nodes:\n"
        )
        prompts.append(prompt)
 
    outputs = []
    print("Gonna Generate samples")
    pbar = tqdm(total=args.num_samples, desc="Generating samples")
    start_time = time.time()
    
    for i in range(0, args.num_samples, args.batch_size):
        batch_size = min(args.batch_size, args.num_samples - i)
        batch_prompts = prompts[i:i+batch_size]

        batch = tokenizer(
            list(batch_prompts), 
            return_tensors="pt",
        )
        batch = {k: v.cuda() for k, v in batch.items()}

        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=2048,
            temperature=args.temperature, 
            top_p=args.top_p, 
        )

        gen_strs = tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        for gen_str, prompt in zip(gen_strs, batch_prompts):
            try:
                material_str = gen_str.replace(prompt, "")
                lengths, angles, coords, edge_index = parse_fn(material_str)
                outputs.append({
                    "lengths": lengths,
                    "angles": angles,
                    "coordinates": coords,
                    "edge_index": edge_index,
                })
            except Exception as e:
                print(e)
            pbar.update(1)
            
    pbar.close()
    calculate_generation_time(start_time, args.num_samples)
    # Save the collected structures to a pickle file
    pkl_file_path = os.path.join(args.out_path, "/home/zhang/Metamaterial-Benchmark/crystal-text-llm/uncond_generated_structures.pkl")
    with open(pkl_file_path, "wb") as pkl_file:
        pickle.dump(outputs, pkl_file)

    print(f"All generated structures saved to {pkl_file_path}")


def conditional_sample(args):
    model, tokenizer = prepare_model_and_tokenizer(args)

    conditions = pickle.load(open("/home/zhang/Metamaterial-Benchmark/unit_cell_catalog/test_set.pkl", "rb"))
    prompts = []
    for d in conditions:
        prompt = "Below is a description of a truss metamaterial.\n"
        prompt += parse_mechanical_properties(d['mech_props'][0])
        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "then the coordinates for each node within the lattice, and the edge connections between nodes:\n"
        )
        prompts.append(prompt)
 
    outputs = []
    num_samples = len(prompts)
    pbar = tqdm(total=num_samples, desc="Generating samples")
    start_time = time.time()
    
    for i in range(0, num_samples, args.batch_size):
        batch_size = min(args.batch_size, num_samples - i)
        batch_prompts = prompts[i:i+batch_size]
        batch_conditions = conditions[i:i+batch_size]
        
        batch = tokenizer(
            list(batch_prompts), 
            return_tensors="pt",
        )
        batch = {k: v.cuda() for k, v in batch.items()}

        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=2048,
            temperature=args.temperature, 
            top_p=args.top_p, 
        )

        gen_strs = tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        for gen_str, prompt, _conditions in zip(gen_strs, batch_prompts, batch_conditions):
            try:
                material_str = gen_str.replace(prompt, "")
                lengths, angles, coords, edge_index = parse_fn(material_str)
                outputs.append({
                    "lengths": lengths,
                    "angles": angles,
                    "coordinates": coords,
                    "edge_index": edge_index,
                    "mech_props": _conditions['mech_props'][0]
                })
            except Exception as e:
                print(e)
            pbar.update(1)
            
    pbar.close()
    calculate_generation_time(start_time, args.num_samples)
    pkl_file_path = os.path.join(args.out_path, "/home/zhang/Metamaterial-Benchmark/crystal-text-llm/cond_generated_structures.pkl")
    with open(pkl_file_path, "wb") as pkl_file:
        pickle.dump(outputs, pkl_file)
    print(f"All generated structures saved to {pkl_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out_path", type=str, default="uncond_samples_oct17")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--format_instruction_prompt", type=int, default=0)
    parser.add_argument("--format_response_format", type=int, default=0)
    parser.add_argument("--conditions", default=False, action="store_true")
    parser.add_argument("--conditions_file", type=str, default="/home/zhang/crystal-text-llm/evaluation_10k.csv") #"data/with_tags/test.csv"
    parser.add_argument("--infill_file", type=str, default="") #"data/with_tags/test.csv"
    parser.add_argument("--infill_do_constraint", type=int, default=0)
    parser.add_argument("--infill_constraint_tolerance", type=float, default=0.1)
    # parser.add_argument("--conditional", type=int, default=0)
    args = parser.parse_args()


    if args.conditions:
        print("Conditional sampling")
        conditional_sample(args)
    else:
        print("Unconditional sampling")
        unconditional_sample(args)