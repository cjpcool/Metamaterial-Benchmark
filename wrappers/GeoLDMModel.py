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
from models.geoldm.qm9.models import get_optim, get_autoencoder, get_latent_diffusion
from models.geoldm.equivariant_diffusion import en_diffusion
from models.geoldm.equivariant_diffusion import utils as flow_utils
from models.geoldm.qm9.sampling import  sample
from models.geoldm.train_test import train_epoch, test, analyze_and_save
from wrappers.BaseModel import BaseModel
from models.geoldm.qm9.utils import prepare_context
from datasets.dataset_truss import LatticeStiffness, LatticeModulus
import datetime

from torch.utils.data import Dataset, DataLoader




class GeolDMLatticeData(Dataset):
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


class GeoLDMModel(BaseModel):

    def __init__(self, model_name, dataset_name, device=torch.device('cuda'), root_path='../'):
        super(GeoLDMModel, self).__init__(model_name, dataset_name, device, root_path)
        self.dtype = torch.float32
        # 其它内部变量初始化
        self.nodes_dist = None
        self.prop_dist = None
        self.gradnorm_queue = None
        self.ema = None
        self.model_ema = None
        self.model_dp = None
        self.dataset_info = None


    def load_model(self, checkpoint=None):
        if self.config["train_diffusion"]:
            self.model, self.nodes_dist, self.prop_dist = get_latent_diffusion(Namespace(**self.config), self.device, self.dataset_info, self.train_loader)
            self.prop_dist.set_normalizer(self.property_norms)
        else:
            self.model, self.nodes_dist, self.prop_dist = get_autoencoder(Namespace(**self.config), self.device, self.dataset_info, self.train_loader)

        if checkpoint is not None:
            flow_path = join(checkpoint, 'flow.npy')
            if os.path.exists(flow_path):
                flow_state_dict = torch.load(flow_path, map_location=self.device)
                self.model.load_state_dict(flow_state_dict)
                print(f"Model loaded from {checkpoint}")
            else:
                print(f"Checkpoint not found at {flow_path}")

    def save_results(self, path='results/'):
        pass

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
        self.train_data, self.val_data, self.test_data = (GeolDMLatticeData(dataset[split_idx['train']]),
                                                          GeolDMLatticeData(dataset[split_idx['valid']]),
                                                          GeolDMLatticeData(dataset[split_idx['test']]))

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


    def train(self):
        property_norms = self.property_norms


        self.model = self.model.to(self.device)

        # 初始化梯度范数队列
        self.gradnorm_queue = utils.Queue()
        self.gradnorm_queue.add(3000)

        # 初始化 EMA 副本（若启用）
        if self.config["ema_decay"] > 0:
            self.model_ema = copy.deepcopy(self.model)
            self.ema = flow_utils.EMA(self.config["ema_decay"])
        else:
            self.ema = None
            self.model_ema = self.model

        if self.config["dp"] and torch.cuda.device_count() > 1:
            print(f"Training using {torch.cuda.device_count()} GPUs")
            self.model_dp = torch.nn.DataParallel(self.model.cpu()).cuda()
        else:
            self.model_dp = self.model

        if not self.config["wandb_args"]["use_wandb"]:
            mode = "disabled"
        else:
            mode = "online" if self.config.get("online", True) else "offline"
        wandb.init(entity=self.config["wandb_args"]["entity"],
                   name=self.config["wandb_args"]["save_name"]+'-'+datetime.datetime.now().strftime('%Y-%m-%d--%H:%M'),
                   project=self.config["wandb_args"]["project"],
                   config=self.config,
                   settings=wandb.Settings(_disable_stats=True),
                   reinit=True,
                   mode=mode)
        wandb.save("*.txt")

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'], amsgrad=True,
            weight_decay=1e-12)

        args = self.config

        self.best_nll_val = 1e8
        self.best_nll_test = 1e8
        for epoch in range(self.config["start_epoch"], self.config["n_epochs"]):
            start_time = time.time()
            train_epoch(args=Namespace(**self.config),
                        loader=self.train_loader,
                        epoch=epoch,
                        model=self.model,
                        model_dp=self.model_dp,
                        model_ema=self.model_ema,
                        ema=self.ema,
                        device=self.device,
                        dtype=self.dtype,
                        property_norms=property_norms,
                        nodes_dist=self.nodes_dist,
                        dataset_info=self.dataset_info,
                        gradnorm_queue=self.gradnorm_queue,
                        optim=self.optim,
                        prop_dist=self.prop_dist)
            end_time = time.time()
            print(f"Epoch {epoch} took {end_time - start_time:.1f} seconds, MTT={(end_time - start_time)/len(self.train_loader)} seconds")
            if epoch % args["test_epochs"] == 0:
                if isinstance(self.model, en_diffusion.EnVariationalDiffusion):
                    wandb.log(self.model.log_info(), commit=True)
                # if (not args["break_train_epoch"]) and args["train_diffusion"]:
                #     analyze_and_save(args=Namespace(**self.config),
                #                      epoch=epoch,
                #                      model_sample=self.model_ema,
                #                      nodes_dist=self.nodes_dist,
                #                      dataset_info=self.dataset_info,
                #                      device=self.device,
                #                      prop_dist=self.prop_dist,
                #                      n_samples=args["n_stability_samples"])
                nll_val = test(args=Namespace(**self.config),
                               loader=self.val_loader,
                               epoch=epoch,
                               eval_model=self.model_ema,
                               partition="Val",
                               device=self.device,
                               dtype=self.dtype,
                               nodes_dist=self.nodes_dist,
                               property_norms=property_norms)
                nll_test = test(args=Namespace(**self.config),
                                loader=self.test_loader,
                                epoch=epoch,
                                eval_model=self.model_ema,
                                partition="Test",
                                device=self.device,
                                dtype=self.dtype,
                                nodes_dist=self.nodes_dist,
                                property_norms=property_norms)
                if nll_val < self.best_nll_val:
                    self.best_nll_val = nll_val
                    self.best_nll_test = nll_test
                    if args["save_model"]:
                        os.makedirs(os.path.join(args['save_dir'], args['exp_name']), exist_ok=True)
                        print("Saving model to", os.path.join(args['save_dir'], args['exp_name']))
                        utils.save_model(self.optim, os.path.join(args['save_dir'], args['exp_name'], 'args.pickle'))
                        utils.save_model(self.model, os.path.join(args['save_dir'], args['exp_name'], 'generative_model.npy'))
                        if args['ema_decay'] > 0:
                            utils.save_model(self.model_ema,
                                             os.path.join(args['save_dir'],args['exp_name'], 'generative_model_ema.npy'))
                        with open(os.path.join(args['save_dir'],args['exp_name'], 'args.pickle'), 'wb') as f:
                            pickle.dump(args, f)
                print(f"Val loss: {nll_val:.4f} Test loss: {nll_test:.4f}")
                print(f"Best val loss: {self.best_nll_val:.4f} Best test loss: {self.best_nll_test:.4f}")
                wandb.log({"Val loss": nll_val}, commit=True)
                wandb.log({"Test loss": nll_test}, commit=True)
                wandb.log({"Best cross-validated test loss": self.best_nll_test}, commit=True)

    def evaluate(self):
        pass

    def test(self):
        pass

    def visualize(self, path='results/plots/'):
        pass

    def generate(self, save_path='gen_results/geoldm/save_results', generators_path=None,  n_sweeps=10,n_frames=100, task='qualitative', ):
        device = self.device
        if generators_path is not None:
            args_gen = get_args_gen(generators_path)
            model, nodes_dist, prop_dist, dataset_info = get_generator(generators_path, self.train_loader, device, args_gen,
                                                                       self.property_norms)
            # args_gen['context_node_nf'] = self.config['context_node_nf']
        else:
            args_gen = self.config
            model, nodes_dist, prop_dist, dataset_info = self.model, self.nodes_dist, self.prop_dist, self.dataset_info

        if task == 'qualitative':
            for i in range(n_sweeps):
                n_nodes = nodes_dist.sample(1).item()
                print(f"Sampling sweep {i + 1}/{n_sweeps}, n_nodes {n_nodes}")
                save_and_sample_conditional(args_gen, device, model, prop_dist, dataset_info, n_nodes=n_nodes, epoch=i, id_from=0, save_path=save_path, batch_num=i, n_frames=n_frames, test_dataset=None)
        else:
            raise ValueError("Only 'qualitative' task is supported in this function.")

def sample_sweep_conditional(args, device, generative_model, dataset_info, prop_dist, n_nodes=19, n_frames=100, test_dataset=None):
    nodesxsample = torch.tensor([n_nodes] * n_frames)

    context = []
    if test_dataset is None:
        for key in prop_dist.distributions:
            min_val, max_val = prop_dist.distributions[key][n_nodes]['params']
            mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
            min_val = (min_val - mean) / (mad)
            max_val = (max_val - mean) / (mad)
            context_row = torch.tensor(np.linspace(min_val, max_val, n_frames)).unsqueeze(1)
            context.append(context_row)
        context = torch.cat(context, dim=1).float().to(device)
    else:
        for key in prop_dist.distributions:
            min_val, max_val = prop_dist.distributions[key][n_nodes]['params']
            mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
            # min_val = (min_val - mean) / (mad)
            # max_val = (max_val - mean) / (mad)
            idx = torch.nonzero(test_dataset.data['num_atoms'] == n_nodes).view(-1).numpy()
            np.random.shuffle(idx)
            while n_frames > len(idx):
                idx = np.concatenate((idx, idx))
            idx = idx[:n_frames]
            context_row = test_dataset.data[key][idx].unsqueeze(1)
            context_row = (context_row - mean) / (mad)
            # context_row = torch.tensor(np.linspace(min_val, max_val, n_frames)).unsqueeze(1)
            context.append(context_row)

        context = torch.cat(context, dim=1).float().to(device)
    one_hot, charges, x, node_mask = sample(args, device, generative_model, dataset_info, prop_dist, nodesxsample=nodesxsample, context=context, fix_noise=True)
    return one_hot, charges, x, node_mask, context

def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, n_nodes, epoch=0, id_from=0, save_path='', batch_num=0, n_frames=100, test_dataset=None):
    start_tiem = time.time()
    one_hot, charges, x, node_mask, context = sample_sweep_conditional(Namespace(**args), device, model, dataset_info, prop_dist, n_nodes=n_nodes,n_frames=n_frames, test_dataset=test_dataset)
    end_tiem = time.time()
    print(f'Generate {n_frames} samples time={end_tiem - start_tiem}s.')

    os.makedirs(save_path, exist_ok=True)

    for i in range(x.shape[0]):
        lattice_name = os.path.join(save_path, f'lattices_{batch_num}_{i}.npy' )
        oht_i = one_hot[i]
        x_i = x[i]
        node_mask_i = node_mask[i].squeeze(-1)== 1.0
        oht_i = oht_i[node_mask_i]
        x_i = x_i[node_mask_i]
        atom_types = torch.argmax(oht_i).cpu().numpy()
        lengths = torch.FloatTensor([1.0,1.0,1.0]).cpu().numpy()
        angels = torch.FloatTensor([90,90,90]).cpu().numpy()
        cart_coords = x_i.cpu().numpy()
        edge_index = None
        context_i = context[i].squeeze(0).cpu()
        # for key in prop_dist.distributions:
        # min_val, max_val = prop_dist.distributions['y'][n_nodes]['params']
        mean, mad = prop_dist.normalizer['y']['mean'], prop_dist.normalizer['y']['mad']
        context_i = context_i*mad + mean
        prop_list = context_i.numpy()
        # prop_list -= prop_dist.

        np.savez(lattice_name,
                 atom_types=atom_types,
                 lengths=lengths,
                 angles=angels,
                 cart_coords=cart_coords,
                 edge_index=edge_index,
                 prop_list=prop_list,
                 )
    return one_hot, charges, x



def get_generator(dir_path, train_loader, device, args_gen, property_norms):
    print('load data')
    if args_gen['data_name'] == 'LatticeModulus':
        dataset_info = LatticeModulus_info
    elif args_gen['data_name'] == 'LatticeStiffness':
        dataset_info = LatticeModulus_info

    model, nodes_dist, prop_dist = get_latent_diffusion(Namespace(**args_gen), device, dataset_info, train_loader)
    fn = 'generative_model_ema.npy' if args_gen['ema_decay'] > 0 else 'generative_model.npy'
    model_state_dict = torch.load(join(dir_path, fn), map_location='cpu')
    model.load_state_dict(model_state_dict)

    # The following function be computes the normalization parameters using the 'valid' partition

    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
    return model.to(device), nodes_dist, prop_dist, dataset_info

def get_args_gen(dir_path):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_gen = pickle.load(f)
    # assert args_gen.dataset == 'qm9_second_half'

    # Add missing args!
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen['normalization_factor'] = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen['aggregation_method'] = 'sum'
    return args_gen