import os
import copy
import time
import pickle
from argparse import Namespace

import torch
import wandb
from os.path import join
# from torch_geometric.loader import DataLoader
import numpy as np

from models.geoldm import utils
from models.geoldm.data_info.datasets_config import get_dataset_info, geom_with_h, LatticeModulus_info
# from models.geoldm.eval_conditional_qm9 import get_generator

# only different from geoldm is the get_model
from models.edm.qm9.models import get_optim, get_model
from models.geoldm.equivariant_diffusion import en_diffusion
from models.geoldm.equivariant_diffusion import utils as flow_utils
from models.geoldm.qm9.sampling import  sample
from models.geoldm.train_test import train_epoch, test, analyze_and_save
from wrappers.BaseModel import BaseModel
from models.geoldm.qm9.utils import prepare_context
from datasets.dataset_truss import LatticeStiffness, LatticeModulus
import datetime

from torch.utils.data import Dataset, DataLoader
from wrappers.GeoLDMModel import GeoLDMModel, sample_sweep_conditional,get_args_gen,save_and_sample_conditional



class EDMLatticeData(Dataset):
    def __init__(self, pyg_data):
        self.data_list = [pyg_data[i] for i in range(len(pyg_data))]
        self.data = {'num_atoms': pyg_data.num_atoms,
                     'y': pyg_data.y,
                     }

        if hasattr(self.data, 'young'):
            self.data['young'] = pyg_data.young
        if hasattr(self.data, 'poisson'):
            self.data['poisson'] = pyg_data.poisson
        if hasattr(self.data, 'shear'):
            self.data['shear'] = pyg_data.shear



        self.max_nodes = max(data.cart_coords.shape[0] for data in self.data_list)


    def __len__(self):
        return len(self.data_list)

    def pad_pos(self, pos, node_feat):
        """
        对 pos 进行 padding
        Args:
            pos (Tensor): (N, D) 形状的节点特征
        Returns:
            padded_pos (Tensor): (max_nodes, D) 形状的节点特征
        """
        N, D = pos.shape
        pad_size = self.max_nodes - N
        D1 = node_feat.shape[1]
        if pad_size > 0:
            pad_tensor = torch.zeros((pad_size, D))  # 创建填充矩阵
            padded_pos = torch.cat([pos, pad_tensor], dim=0)  # 在行方向拼接
            pad_tensor_x = torch.zeros((pad_size, D1))
            padded_feat = torch.cat([node_feat, pad_tensor_x], dim=0)  # 在行方向拼接
        else:
            padded_pos = pos
            padded_feat = node_feat

        return padded_pos, padded_feat



    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data_list[idx]
        n = data.num_atoms[0]

        padded_pos, node_feat = self.pad_pos(data.cart_coords, data.node_feat)
        # i,j = data.edge_index
        atom_mask = torch.zeros(padded_pos.shape[0])
        atom_mask[:n] = 1
        atom_mask = atom_mask == 1.
        # edge_index = torch.zeros((n, n))
        # edge_index[i,j] = 1

        edge_mask = torch.ones((padded_pos.shape[0], padded_pos.shape[0]))
        edge_mask[torch.eye(edge_mask.shape[0])==0.] = 0

        new_data = {
            "positions": padded_pos,  # (max_nodes, D)
            'one_hot': node_feat,
            "num_atoms": data.num_atoms,  # 保持 num_atoms
            # "edge_index": edge_index.flatten(),  # 关系图结构不变
            "y": data.y,
            'charges': torch.zeros(0),
            'atom_mask': atom_mask,
            'edge_mask':  edge_mask.flatten().bool()
        }
        if hasattr(data, 'young'):
            new_data['young'] = data.young
        if hasattr(data, 'poisson'):
            new_data['poisson'] = data.poisson
        if hasattr(data, 'shear'):
            new_data['shear'] = data.shear

        return new_data


class EDMModel(GeoLDMModel):


    def load_model(self, checkpoint=None):
        self.model, self.nodes_dist, self.prop_dist = get_model(Namespace(**self.config), self.device,
                                                                           self.dataset_info, self.train_loader)
        self.prop_dist.set_normalizer(self.property_norms)

        if checkpoint is not None:
            flow_path = join(checkpoint, 'flow.npy')
            if os.path.exists(flow_path):
                flow_state_dict = torch.load(flow_path, map_location=self.device)
                self.model.load_state_dict(flow_state_dict)
                print(f"Model loaded from {checkpoint}")
            else:
                print(f"Checkpoint not found at {flow_path}")

    def load_data(self):
        print('load data')
        if self.config['data_name'] == 'LatticeModulus':
            self.dataset_info = LatticeModulus_info
        elif self.config['data_name'] == 'LatticeStiffness':
            self.dataset_info = LatticeModulus_info

        dataset = eval("{}('{}')".format(self.config['data_name'], self.config['data_path'], encoding='utf-8'))
        idx = torch.nonzero(dataset.num_atoms < self.dataset_info['max_n_nodes'])
        dataset = dataset.copy(idx)

        split_idx = dataset.get_idx_split(len(dataset), train_size=self.config['train_size'],
                                          valid_size=self.config['valid_size'], seed=self.config['data_seed'])
        print(split_idx.keys())
        print(dataset[split_idx['train']])
        self.train_data, self.val_data, self.test_data = (EDMLatticeData(dataset[split_idx['train']]),
                                                          EDMLatticeData(dataset[split_idx['valid']]),
                                                          EDMLatticeData(dataset[split_idx['test']]))

        self.train_loader = DataLoader(self.train_data, batch_size=self.config["batch_size"], shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=self.config["val_batch_size"], shuffle=False)
        self.val_loader = DataLoader(self.val_data, batch_size=self.config["val_batch_size"], shuffle=False)


        if self.prop_dist is not None:
            self.prop_dist.set_normalizer(self.config.get("property_norms", None))

        context_node_nf = 0
        self.property_norms = {}
        data_dummy = next(iter(self.train_loader))
        if len(self.config['conditioning']) > 0:
            print(f'Conditioning on {self.config["conditioning"]}')
            for property_key in self.config['conditioning']:
                values = self.train_data.data[property_key]
                mean = torch.mean(values, dim=0)
                ma = torch.abs(values - mean)
                mad = torch.mean(ma, dim=0)
                self.property_norms[property_key] = {}
                self.property_norms[property_key]['mean'] = mean
                self.property_norms[property_key]['mad'] = mad
                context_dummy = prepare_context(self.config['conditioning'], data_dummy, self.property_norms)
                context_node_nf = context_dummy.size(2)
        self.config['context_node_nf'] = context_node_nf



        return self.train_data, self.val_data, self.test_data

    def generate(self, save_path='gen_results/edm/save_results', generators_path=None,  n_sweeps=10,n_frames=100, task='qualitative', use_test_data=False):
        device = self.device
        if generators_path is not None:
            args_gen = get_args_gen(generators_path)
            model, nodes_dist, prop_dist, dataset_info = get_generator(generators_path, self.train_loader, device, args_gen,
                                                                       self.property_norms)
            # args_gen['context_node_nf'] = self.config['context_node_nf']
        else:
            args_gen = self.config
            model, nodes_dist, prop_dist, dataset_info = self.model, self.nodes_dist, self.prop_dist, self.dataset_info

        if use_test_data:
            test_data = self.test_data

        if task == 'qualitative':
            for i in range(n_sweeps):
                n_nodes = nodes_dist.sample(1).item()
                print(f"Sampling sweep {i + 1}/{n_sweeps}, n_nodes {n_nodes}")
                save_and_sample_conditional(args_gen, device, model, prop_dist, dataset_info, n_nodes=n_nodes, epoch=i, id_from=0, save_path=save_path, batch_num=i, n_frames=n_frames, test_dataset=test_data)
        else:
            raise ValueError("Only 'qualitative' task is supported in this function.")


def get_generator(dir_path, train_loader, device, args_gen, property_norms):
    print('load data')
    if args_gen['data_name'] == 'LatticeModulus':
        dataset_info = LatticeModulus_info
    elif args_gen['data_name'] == 'LatticeStiffness':
        dataset_info = LatticeModulus_info

    model, nodes_dist, prop_dist = get_model(Namespace(**args_gen), device, dataset_info, train_loader)
    fn = 'generative_model_ema.npy' if args_gen['ema_decay'] > 0 else 'generative_model.npy'
    model_state_dict = torch.load(join(dir_path, fn), map_location='cpu')
    model.load_state_dict(model_state_dict)

    # The following function be computes the normalization parameters using the 'valid' partition

    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
    return model.to(device), nodes_dist, prop_dist, dataset_info
