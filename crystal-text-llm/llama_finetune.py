"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import glob
import argparse
import torch
import random
import warnings
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
from pathlib import Path
import random

from dataclasses import dataclass
import transformers
from transformers import ( 
    LlamaForCausalLM,
    LlamaTokenizer, 
    Trainer, 
    TrainingArguments
)

from torch.utils.data import Dataset

from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
import pickle

IGNORE_INDEX = -100
MAX_LENGTH = 2048
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


def get_lattice_string(lengths, angles, nodes, edges):
    # Format lengths and angles into strings
    lengths_str = f"{lengths[0][0]:.4f}, {lengths[0][1]:.4f}, {lengths[0][2]:.4f}"
    angles_str = f"{angles[0][0]:.1f}, {angles[0][1]:.1f}, {angles[0][2]:.1f}"
    # Combine into header text
    header = f"{lengths_str}\n{angles_str}"

    coords_list = nodes
    coords_rows = [f"{i} " + ' '.join([f"{coord:.3f}" for coord in row]) for i, row in enumerate(coords_list)]
    coords_text = "\n".join(coords_rows)

    edge_index_list = edges
    edge_index_rows = [' '.join([f"{int(index)}" for index in row]) for row in edge_index_list]
    edge_index_text = "\n".join(edge_index_rows)

    return header + "\n\n" + coords_text + "\n\n" + edge_index_text

def get_nodes_string(nodes):
    coords_list = nodes
    coords_rows = [f"{i} " + ' '.join([f"{coord:.3f}" for coord in row]) for i, row in enumerate(coords_list)]
    coords_text = "\n".join(coords_rows)
    return coords_text

def get_edges_string(edges):
    edge_index_list = edges
    edge_index_rows = [' '.join([f"{int(index)}" for index in row]) for row in edge_index_list]
    edge_index_text = "\n".join(edge_index_rows)
    return edge_index_text


def parse_mechanical_properties(sequence):
    
    properties = {
        "Ex": sequence[0],
        "Ey": sequence[1],
        "Ez": sequence[2],
        "Gyz": sequence[3],
        "Gxz": sequence[4],
        "Gxy": sequence[5],
        "nuyz": sequence[6],
        "nuxz": sequence[7],
        "nuxy": sequence[8],
        "nuzy": sequence[9],
        "nuzx": sequence[10],
        "nuyx": sequence[11],
    }

    output_text = (
        f"Effective normalized mechanical properties:\n"
        f"Ex  = {properties['Ex']:.2E}, Ey  = {properties['Ey']:.2E}, Ez  = {properties['Ez']:.2E}\n"
        f"Gyz = {properties['Gyz']:.2E}, Gxz = {properties['Gxz']:.2E}, Gxy = {properties['Gxy']:.2E}\n"
        f"nuyz = {properties['nuyz']:.3f}, nuxz = {properties['nuxz']:.3f}, nuxy = {properties['nuxy']:.3f}, "
        f"nuzy = {properties['nuzy']:.3f}, nuzx = {properties['nuzx']:.3f}, nuyx = {properties['nuyx']:.3f}"
    )

    return output_text


class LatticeDataset(Dataset):
    def __init__(
        self,
        lattices,
        llama_tokenizer=None,
        w_attributes=False,
    ):
        super().__init__()

        self.inputs = lattices
        self.llama_tokenizer = llama_tokenizer
        self.w_attributes = w_attributes

    def generation_task(self, input_dict):

        prompt = "Below is a description of a truss metamaterial.\n"
        
        if self.w_attributes and random.uniform(-1, 1) > 0:
            prompt += parse_mechanical_properties(input_dict['mech_props'][0])
    
        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "then the coordinates for each node within the lattice, and the edge connections between nodes:\n"
        )

        lattice_str = get_lattice_string(input_dict['lengths'], input_dict['angles'], input_dict['nodes'], input_dict['edges'])
        # print(lattice_str)
        tokens = self.llama_tokenizer(
            prompt + lattice_str  + self.llama_tokenizer.eos_token,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        return tokens


    def tokenize(self, input_dict):
        
        tokens = self.generation_task(input_dict)

        input_ids = labels = tokens.input_ids[0]
        input_ids_lens = labels_lens = tokens.input_ids.ne(
            self.llama_tokenizer.pad_token_id).sum().item()
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")

        vals = self.inputs[index]
        input_dict = {
            "lengths": vals['lengths'],
            "angles": vals['angles'],
            "nodes": vals['frac_coords'],
            "edges": vals['edge_index'].T,
            "mech_props": vals['mech_props'],
        }
        vals = self.tokenize(input_dict)
        return vals

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        # print(instances)
        input_ids, labels = tuple(
            [instance[key].clone().detach() for instance in instances] 
                for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def setup_datasets(args, llama_tokenizer, lattices_path = "LatticeModulus", transform_args={}):    
    try:
        print(f"Attempting to load dataset from path: {lattices_path}")
        train_catalog = pickle.load(open("unit_cell_catalog/train_set.pkl", "rb"))
        val_catalog = pickle.load(open("unit_cell_catalog/val_set.pkl", "rb"))
        test_catalog = pickle.load(open("unit_cell_catalog/test_set.pkl", "rb"))
        
        all_datasets = {
            "train": LatticeDataset(
                lattices=train_catalog,
                llama_tokenizer=llama_tokenizer,
                w_attributes=args.w_attributes,
            ),
            "val": LatticeDataset(
                lattices=val_catalog,
                llama_tokenizer=llama_tokenizer,
                w_attributes=args.w_attributes,
            ),
            "test": LatticeDataset(
                lattices=test_catalog,
                llama_tokenizer=llama_tokenizer,
                w_attributes=args.w_attributes,
            )
        }
        
        return all_datasets
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print(f"Please verify that the dataset exists at: {lattices_path}")
        raise


def setup_training_args(args):
    output_dir= args.expdir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        os.environ["WANDB_DISABLED"] = "True"
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
    training_args = TrainingArguments(
        fsdp=False,
        fp16=not args.fp8,
        bf16=False,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        num_train_epochs=args.num_epochs,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=10,
        evaluation_strategy="no",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.num_warmup_steps,
        # warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=output_dir,
        run_name=args.run_name,
        report_to="wandb",
        dataloader_num_workers=8,
        remove_unused_columns=False,
        label_names=["crystal_ids"], #this is just to get trainer to behave how I want
    )
    return training_args

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

def setup_model(args, rank):
    llama_options = args.model_name.split("-")
    is_chat = len(llama_options) == 2
    model_size = llama_options[0]

    def llama2_model_string(model_size, chat):
        chat = "chat-" if chat else ""
        return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

    model_string = llama2_model_string(model_size, is_chat)

    model = LlamaForCausalLM.from_pretrained(
        model_string,
        load_in_8bit=args.fp8,
        device_map={"": rank},
    )

    llama_tokenizer = LlamaTokenizer.from_pretrained(
        model_string,
        model_max_length=MAX_LENGTH,
        padding_side="right",
        use_fast=False,
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    special_tokens_dict = dict()
    if llama_tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if llama_tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if llama_tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if llama_tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=llama_tokenizer,
        model=model,
    )

    return model, llama_tokenizer

def setup_trainer(args):
    training_args = setup_training_args(args)
    model, llama_tokenizer = setup_model(args, training_args.local_rank)

    all_datasets = setup_datasets(args, llama_tokenizer)

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=llama_tokenizer, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=all_datasets["train"],
        eval_dataset=all_datasets["val"],
        data_collator=data_collator,
    )

    return trainer

def main(args):
    trainer = setup_trainer(args)

    if args.resume_dir is not None:
        train_result = trainer.train(resume_from_checkpoint=args.resume_dir)
    else:
        train_result = trainer.train()

    print(train_result)
    trainer.save_state()
    trainer.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--expdir", type=Path, default="crystal-text-llm/exp")
    parser.add_argument("--model_name", default="7b")
    parser.add_argument("--fp8", action="store_true", default=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--data-path", type=Path, default="data/basic")
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--num-warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-freq", default=1000, type=int)
    parser.add_argument("--save-freq", default=5000, type=int)
    parser.add_argument("--format-permute-composition", action="store_true", default=False)
    parser.add_argument("--format-permute-structure", action="store_true", default=False)
    parser.add_argument("--w-attributes", type=int, default=1)
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    main(args)