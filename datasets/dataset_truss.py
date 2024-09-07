import os
from typing import List, Callable

from sklearn.utils import shuffle

import torch
import pandas as pd
# from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from datasets.lattice import Structure
from utils.lattice_utils import Topology
import pickle

from torch_geometric.datasets import MD17
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
    extract_zip,
)

class LatticeModulus(InMemoryDataset):
    downloadurl = ''

    file_names = {'LatticeModulus':'data.pkl'}
    name = 'LatticeModulus'
    def __init__(self, data_path: str,
                 file_name='data'):
        # define column names from data
        self.raw_data_keys = ['Name', 'Other name(s)', 'lengths', 'angles', 'Z_avg', 'Young', 'Shear', 'Poisson',
                              'Emax',
                              'Scaling constants', 'Scaling exponents', 'has overlapping bars', 'Nodal positions',
                              'Edge index']
        self.properties_names = ['Young', 'Shear', 'Poisson']

        self.data_path = data_path
        self.file_name = file_name

        super().__init__(root=data_path)

        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_data_exist(self):
        return os.path.exists(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.file_name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.file_name, 'processed')
    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    @property
    def raw_file_names(self) -> str:
        name = self.file_names[self.name]
        return name

    def download(self):
        url = self.downloadurl
        path = download_url(url, self.raw_dir)
        return path


    def process(self):
        print('Processing data...')
        with open(self.raw_paths[0], mode='rb') as f:
            raw_data = pickle.load(f)

        data_list = []
        #['Name', 'Other name(s)', 'lengths', 'angles', 'Z_avg', 'Young', 'Shear', 'Poisson', 'Emax',
         # 'Scaling constants', 'Scaling exponents', 'has overlapping bars', 'Nodal positions', 'Edge index']
        for i in tqdm(range(len(raw_data))):
        # for i in tqdm(range(1000)):
            exported_lattice = raw_data[i]
            try:
                properties = torch.FloatTensor(exported_lattice['Young']+exported_lattice['Shear']+exported_lattice['Poisson'])
                lattice_param = torch.FloatTensor([exported_lattice['lengths'],exported_lattice['angles']])
                frac_coords = torch.FloatTensor(exported_lattice['Nodal positions']).squeeze(1) - torch.tensor([0.5,0.5,0.5])
                cart_coords = Structure.frac_to_cart_coords(frac_coords, lattice_param[0], lattice_param[1], frac_coords.shape[0])
                edge_index = torch.LongTensor(exported_lattice['Edge index']).squeeze(1)
                # "tol" control the radius: If the nodes' distances smaller than this radius, then only one of them remains.
                edge_index, cart_coords = Structure.remove_overlapping_nodes(edge_index.numpy(), cart_coords.numpy(), tol=1e-5)

                S1 = Structure(lattice_param,
                               edge_index,
                               cart_coords, is_cartesian=True,
                               properties=properties,
                               properties_names=self.properties_names)
            except KeyError as e:
                print('Error sample {}, skip it.'.format(i), e)
                continue
            edge_num = S1.num_edges
            node_feat = torch.zeros((S1.num_nodes, 1), dtype=torch.long)
            edge_feat = torch.zeros((edge_num, 1), dtype=torch.float32)

            data = Data(
                frac_coords=S1.frac_coords.to(torch.float32),
                cart_coords=S1.cart_coords.to(torch.float32),
                node_feat=node_feat,
                edge_feat=edge_feat,
                edge_index=S1.edge_index,
                num_nodes=S1.num_nodes,
                num_atoms=S1.num_nodes,
                num_edges=edge_num,
                lengths=S1.lattice_params[0].view(1, -1).to(torch.float32),
                angles=S1.lattice_params[1].view(1, -1).to(torch.float32),
                vector=S1.lattice_vector.view(1, -1),
                y=S1.properties.to(torch.float32),
                to_jimages = S1.to_jimages
            )



            data_list.append(data)
        print('End preprocessing data.')
        print('Saving data...')
        torch.save(self.collate(data_list), self.processed_paths[0])
        print('Completed preprocessing data.')


    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.LongTensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict




class LatticeStiffness(InMemoryDataset):
    # define column names from data
    F1_features_names = ['relative_density', 'U1', 'U2', 'U3', 'lattice_type1', 'lattice_type2', 'lattice_type3',
                         'lattice_rep1', 'lattice_rep2', 'lattice_rep3']
    R1_names = ['R1_theta', 'R1_rot_ax1', 'R1_rot_ax2']
    V_names = ['V1', 'V2', 'V3']
    R2_names = ['R2_theta', 'R2_rot_ax1', 'R2_rot_ax2']
    C_ort_names = ['C11_ort', 'C12_ort', 'C13_ort', 'C22_ort', 'C23_ort', 'C33_ort', 'C44_ort', 'C55_ort',
                   'C66_ort']
    C_names = ['C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C22', 'C23', 'C24', 'C25', 'C26', 'C33', 'C34', 'C35',
               'C36', 'C44', 'C45', 'C46', 'C55', 'C56', 'C66']

    F1_features_scaling_strategy = 'none'
    V_scaling_strategy = 'none'
    C_ort_scaling_strategy = 'none'
    C_scaling_strategy = 'none'
    C_hat_scaling_strategy = 'none'

    downloadurl='https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/520254/training.csv?sequence=1&isAllowed=y'
    file_names = {'LatticeStiffness':'training.csv'}
    name = 'LatticeStiffness'
    def __init__(self, data_path: str,
                 file_name='training'):
        self.data_path = data_path
        self.file_name = file_name

        super().__init__(root=data_path)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_data_exist(self):
        return os.path.exists(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.file_name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.file_name, 'processed')
    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    @property
    def raw_file_names(self) -> str:
        name = self.file_names[self.name]
        return name

    def download(self):
        url = self.downloadurl
        path = download_url(url, self.raw_dir)
        return path


    def process(self):
        print('Processing data...')

        df = pd.read_csv(self.raw_paths[0])
        print(len(df))
        data_list = []

        # for i in tqdm(range(len(df))):
        for i in tqdm(range(1000)):
            dfi = df.iloc[i]
            exported_lattice = Topology(dfi)

            coords = torch.from_numpy(exported_lattice.coordinates)
            lattice_vector = torch.from_numpy(exported_lattice.lattice_vector)
            lengths, angles = Structure.lattice_vector_to_parameters(lattice_vector)

            frac_coords = Structure.cart_to_frac_coords(coords, lengths, angles, coords.shape[0])
        
            if frac_coords.max() > 0.5 or frac_coords.min() < -0.5: # if lattice exceeds the unit cell
                ## if you want to remove these data. However, almost all data has the problem.
                # ignored_num += 1
                # if i %100 == 0:
                #     print('Ignored lattices number: {}'.format(ignored_num))
                # continue
                ## We scale it instead of remove directly.
                frac_coords = frac_coords - frac_coords.min()
                frac_coords = frac_coords / max(frac_coords.max(), 1.)
                frac_coords -= (frac_coords.max(dim=0)[0] + frac_coords.min(dim=0)[0]) / 2.

            S1 = Structure(lattice_vector,
                           torch.from_numpy(exported_lattice.connectity),
                           frac_coords, is_cartesian=False,
                           diameter=exported_lattice.diameter,
                           properties=torch.from_numpy(dfi[self.C_names].values),
                           properties_names=self.C_names)


            edge_num = S1.num_edges
            node_feat = torch.zeros((S1.num_nodes, 1), dtype=torch.long)
            edge_feat = torch.ones((edge_num, 1), dtype=torch.float32) * S1.diameter
            lattice_vector = S1.lattice_vector.view(1, -1)
            data = Data(
                frac_coords=S1.frac_coords.to(torch.float32),
                cart_coords=S1.cart_coords.to(torch.float32),
                node_feat=node_feat,
                edge_feat=edge_feat,
                edge_index=S1.edge_index,
                num_nodes=S1.num_nodes,
                num_atoms=S1.num_nodes,
                num_edges=edge_num,
                lengths=S1.lattice_params[0].view(1, -1).to(torch.float32),
                angles=S1.lattice_params[1].view(1, -1).to(torch.float32),
                vector=lattice_vector.to(torch.float32),
                y=S1.properties.to(torch.float32),
                to_jimages = S1.to_jimages
            )
            data_list.append(data)
        print('End preprocessing data.')
        print('Saving data...')
        torch.save(self.collate(data_list), self.processed_paths[0])
        print('Completed preprocessing data.')


    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.LongTensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict



if __name__ == '__main__':
    from torch_geometric.loader import DataLoader
    from utils.lattice_utils import plot_lattice

    dataset = LatticeStiffness('D:\项目\Material design\code_data\data\LatticeStiffness', file_name='training')
    # dataset = LatticeStiffness('/home/jianpengc/datasets/metamaterial/LatticeStiffness', file_name='training_node_num9')
    # dataset = LatticeModulus('D:\项目\Material design\code_data\data\LatticeModulus', file_name='data')
    split_idx = dataset.get_idx_split(len(dataset), train_size=5, valid_size=5, seed=42)
    print((dataset.frac_coords > 1.0).sum())
    print(split_idx.keys())
    print(dataset[split_idx['train']])
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[
        split_idx['test']]
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    print(dataset)
    data_batch = next(iter(train_loader))
    print(data_batch)
    node_num_per_lattice = dataset.slices['node_feat'][1:] - dataset.slices['node_feat'][:-1]
    edge_num_per_lattice = dataset.num_edges
    print('max node num', max(node_num_per_lattice))
    print('min node num', min(node_num_per_lattice))
    print('average node num', node_num_per_lattice.float().mean())

    print('max edge num', max(edge_num_per_lattice))
    print('average edge num', edge_num_per_lattice.float().mean())
    print('min edge num', min(edge_num_per_lattice))

    data = dataset[10]
    print(data.edge_index.dtype)
    print(data.y)
    print(data.cart_coords.dtype)

    plot_lattice(data.cart_coords, data.edge_index.t())

    from utils.mat_utils import correct_cart_coords, frac_to_cart_coords
    # correct = correct_cart_coords(data.cart_coords,data.lengths, data.angles, data.num_nodes, torch.zeros(data.cart_coords.shape[0], dtype=torch.long))
    plot_lattice(data.frac_coords, data.edge_index.t())

