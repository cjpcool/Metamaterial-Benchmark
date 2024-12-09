import torch
import pandas as pd
from tqdm import tqdm

from datasets.lattice import Structure
from utils.lattice_utils import Topology, scale_to_cell
import pickle

from torch_geometric.data import Data
from datasets.baseDataset import LatticeTruss


class LatticeModulus(LatticeTruss):
    raw_data_keys = ['Name', 'Other name(s)', 'lengths', 'angles', 'Z_avg', 'Young', 'Shear', 'Poisson',
                     'Emax',
                     'Scaling constants', 'Scaling exponents', 'has overlapping bars', 'Nodal positions',
                     'Edge index']

    properties_names = ['Young', 'Shear', 'Poisson']

    downloadurl = ''
    file_names = {'LatticeModulus': 'data.pkl'}
    name = 'LatticeModulus'

    def __init__(self, data_path: str,
                 file_name='data'):

        super().__init__(data_path, file_name)

    def process(self):
        print('Processing data...')
        with open(self.raw_paths[0], mode='rb') as f:
            raw_data = pickle.load(f)

        data_list = []
        # ['Name', 'Other name(s)', 'lengths', 'angles', 'Z_avg', 'Young', 'Shear', 'Poisson', 'Emax',
        # 'Scaling constants', 'Scaling exponents', 'has overlapping bars', 'Nodal positions', 'Edge index']
        for i in tqdm(range(len(raw_data))):
            exported_lattice = raw_data[i]
            try:
                properties = torch.FloatTensor(
                    exported_lattice['Young'] + exported_lattice['Shear'] + exported_lattice['Poisson'])

                S1 = Structure(torch.FloatTensor([exported_lattice['lengths'], exported_lattice['angles']]),
                               torch.LongTensor(exported_lattice['Edge index']).squeeze(1),
                               torch.FloatTensor(exported_lattice['Nodal positions']).squeeze(1), is_cartesian=False,
                               properties=properties,
                               properties_names=self.properties_names)
                # filter data that contains node clusters.
                cart_coords_temp = S1.cart_coords.to(torch.float32)
                dist_mat = torch.cdist(cart_coords_temp, cart_coords_temp) + torch.eye(cart_coords_temp.shape[0], dtype=torch.float32)
                if torch.any(dist_mat < 1e-5):
                    print('Error sample {}, close distance. skip it.'.format(i))
                    continue
            except KeyError:
                print('Error sample {}, skip it.'.format(i))
                continue
            edge_num = S1.edge_index.shape[1]
            node_feat = torch.zeros((S1.num_nodes, 1), dtype=torch.long)
            edge_feat = torch.zeros((edge_num, 1), dtype=torch.float32)
            edge_num = S1.num_edges

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
                vector=S1.lattice_vector.view(1, -1).to(torch.float32),
                y=S1.properties.to(torch.float32).view(1, -1),
                to_jimages=S1.to_jimages
            )

            data_list.append(data)
        print('End preprocessing data.')
        print('Saving data...')
        torch.save(self.collate(data_list), self.processed_paths[0])
        print('Completed preprocessing data.')


class LatticeStiffness(LatticeTruss):
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

    downloadurl = 'https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/520254/training.csv?sequence=1&isAllowed=y'
    file_names = {'LatticeStiffness': 'training.csv'}
    name = 'LatticeStiffness'

    def __init__(self, data_path: str,
                 file_name='training'):
        super().__init__(data_path, file_name)

    def process(self):
        print('Processing data...')

        df = pd.read_csv(self.raw_paths[0])
        print(len(df))
        data_list = []

        for i in tqdm(range(len(df))):
        # for i in tqdm(range(100)):
            dfi = df.iloc[i]
            exported_lattice = Topology(dfi)

            # if exported_lattice.coordinates.shape[0] > 15: continue
            coords = torch.from_numpy(exported_lattice.coordinates)
            # lattice_vector = Structure.find_lattice_vectors(coords)
            lattice_vector = torch.from_numpy(exported_lattice.lattice_vector)
            S1 = Structure(lattice_vector,
                           torch.from_numpy(exported_lattice.connectity),
                           coords, is_cartesian=True,
                           diameter=exported_lattice.diameter,
                           properties=torch.from_numpy(dfi[self.C_names].values),
                           properties_names=self.C_names)
            # if S1.num_nodes != 15: continue
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
                y=S1.properties.to(torch.float32).view(1, -1),
                to_jimages=S1.to_jimages
            )
            # print(data.cart_coords)
            # input()
            data_list.append(data)
        print('End preprocessing data.')
        print('Saving data...')
        print('Sample amount: ' + str(len(data_list)))
        torch.save(self.collate(data_list), self.processed_paths[0])
        print('Completed preprocessing data.')


def main():
    from torch_geometric.loader import DataLoader
    from utils.lattice_utils import plot_lattice

    dataset = LatticeModulus('D:\项目\Material design\code_data\data\LatticeModulus')

    split_idx = dataset.get_idx_split(len(dataset), train_size=5, valid_size=5, seed=42)
    print(split_idx.keys())
    print(dataset[split_idx['train']])
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[
        split_idx['test']]
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    print(dataset)
    data_batch = next(iter(train_loader))
    print('Property shape: ', data_batch.y.shape)
    node_num_per_lattice = dataset.slices['node_feat'][1:] - dataset.slices['node_feat'][:-1]
    edge_num_per_lattice = dataset.num_edges
    print('max node num', max(node_num_per_lattice))
    print('min node num', min(node_num_per_lattice))
    print('average node num', node_num_per_lattice.float().mean())

    print('max edge num', max(edge_num_per_lattice))
    print('average edge num', edge_num_per_lattice.float().mean())
    print('min edge num', min(edge_num_per_lattice))

    data = dataset[0]
    print(data.edge_index.dtype)
    print(data.y)
    print(data.cart_coords.dtype)


if __name__ == '__main__':
    main()
    # plot_lattice(data.cart_coords, data.edge_index.t())