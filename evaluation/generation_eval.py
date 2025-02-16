import os
import sys
import warnings

from scipy.spatial.distance import cdist
from torch_cluster import radius_graph
from tqdm import tqdm

from datasets.dataset_truss import LatticeStiffness, LatticeModulus
from scipy.optimize import linear_sum_assignment
from eval_utils import cart_to_frac_coords
sys.path.append('../')
from typing import List

import torch
import numpy as np

import networkx as nx
from scipy.spatial import distance
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from datasets.baseDataset import LatticeTruss
from eval_utils import lattice_params_to_matrix, frac_to_cart_coords
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.utils import shuffle


def pad_to_max_length(fps_list,max_length, padding_value=float('inf')):
    # Pad each fingerprint to the max_length with the specified padding_value
    padded_fps = [np.pad(fp, ((0,max_length - len(fp)), (0,0)), 'constant', constant_values=(padding_value)) for fp in fps_list]
    return np.array(padded_fps)


class LatticeEvaluatorMaster():
    def __init__(self,
                 cart_coords: List[np.ndarray] = None,
                 frac_coords: List[np.ndarray] = None,
                 node_types: List[np.ndarray] = None,
                 edges: List[np.ndarray] = None,
                 lattice_vectors: List[np.ndarray] = None,
                 ):
        self.cart_coords = [] if cart_coords is None else cart_coords
        self.frac_coords = [] if frac_coords is None else frac_coords
        self.node_types = [] if node_types is None else node_types
        self.edges = [] if edges is None else edges
        self.lattice_vectors = [] if lattice_vectors is None else lattice_vectors


    def eval_graph_validity(self, **kwargs):
        return NotImplementedError

    def eval_condition_effectiveness(self, **kwargs):
        return NotImplementedError

    def obtain_stiffness(self, **kwargs):
        return NotImplementedError

    def eval_diversity(self):
        return NotImplementedError




class LatticeEvaluator(LatticeEvaluatorMaster):
    def __init__(self,
                 test_datset: LatticeTruss=None,
                 eval_file_path: str=None,
                 cart_coords: List[np.ndarray] = None,
                 frac_coords: List[np.ndarray] = None,
                 node_types: List[np.ndarray] = None,
                 edges: List[np.ndarray] = None,
                 lattice_vectors: List[np.ndarray] = None,
                 cluster_size: int = 50,
                 data_size_for_eval: int = 2000,
                 central_symmetry_error_bar=0.1,
                 periodic_error_bar=0.1,
                 diversity_error_bar=0.2):

        super().__init__(
            cart_coords,
            frac_coords,
            node_types,
            edges,
            lattice_vectors
        )
        self.cond_prop = []
        if eval_file_path is not None:
            self.__read_eval_data(eval_file_path)
        else:
            self.cart_coords = cart_coords
            self.frac_coords = frac_coords
            self.node_types = node_types
            self.edges = edges
            self.lattice_vectors = lattice_vectors




        self.central_symmetry_error_bar = central_symmetry_error_bar
        self.periodic_error_bar = periodic_error_bar
        self.diversity_error_bar = diversity_error_bar



        self.test_dataset = test_datset
        self.cluster_size = cluster_size
        if data_size_for_eval is not None:
            self.data_size_for_eval = data_size_for_eval
        else:
            self.data_size_for_eval = len(test_datset)
        assert self.cluster_size <= self.data_size_for_eval, 'data_size_for_eval < cluster_size'


    def __read_eval_data(self, eval_file_path):
        file_names = os.listdir(eval_file_path)
        for file_name in file_names:
            full_path = os.path.join(eval_file_path, file_name)
            lattice_npz = np.load(full_path, allow_pickle=True)


            lattice_lengths = lattice_npz['lengths']
            lattice_angles = lattice_npz['angles']
            try:
                lattice_vector = lattice_npz['vector']
            except:
                lattice_vector = lattice_params_to_matrix(lattice_lengths[0],lattice_lengths[1],lattice_lengths[2],
                                                  lattice_angles[0], lattice_angles[1], lattice_angles[2])
            self.lattice_vectors.append(lattice_vector)

            try:
                frac_coord = lattice_npz['frac_coords']
            except:
                cart_coord = lattice_npz['cart_coords']
                frac_coord = cart_to_frac_coords(cart_coord,lattice_vector, len(cart_coord))

            self.frac_coords.append(frac_coord)

            num_atoms = len(frac_coord)

            try:
                cart_coord = lattice_npz['cart_coords']
            except:
                cart_coord = frac_to_cart_coords(frac_coord,
                                    lattice_vector,
                                    num_atoms)
            self.cart_coords.append(cart_coord)

            atom_types = lattice_npz['atom_types']
            # Removed unused variable declaration
            edge_index = lattice_npz['edge_index']
            # if edge_index is None or np.any(edge_index == np.array(None)):
            #     edge_index = radius_graph(torch.from_numpy(frac_coord), r=1.0, loop=False).numpy()
            self.edges.append(edge_index)

            self.node_types.append(atom_types)

            try:
                cond_prop = lattice_npz['prop_list']
                self.cond_prop.append(cond_prop)
            except:
                print('No conditions')
                pass


    def evaluate_all_uncondition_generation(self):
        periodicity_ratio, mean_symmetry, connectivity_ratio, dangling_node_ratio = self.eval_graph_validity()
        cov_r, cov_p = self.eval_diversity()


    def eval_condition_effectiveness(self):
        eval_number = len(self.cart_coords)
        distances = []
        assert len(self.cart_coords) == len(self.cond_prop)


        idx_node_num  = list(range(len(self.test_dataset)))
        right = self.data_size_for_eval
        selected_idx = shuffle(idx_node_num)[:right]
        selected_data_for_eval = self.test_dataset.copy(selected_idx)

        train_x = [data.cart_coords for data in selected_data_for_eval]
        train_y = test_data.data.y[selected_idx]

        neigh_y = NearestNeighbors(n_neighbors=self.cluster_size, metric='euclidean')
        neigh_y.fit(train_y)


        for i in tqdm(range(eval_number), desc='eval_condition_effectiveness'):
            x_gen = self.cart_coords[i]
            y_cond = self.cond_prop[i]

            dist = self.condition_effectiveness(y_cond, x_gen, train_x, neigh_y)
            # print(dist)

            distances.append(dist)
        print(f'Conditional Effectiveness:{np.mean(distances)}')


    @staticmethod
    def condition_effectiveness(y_cond, x_gen, train_x, neigh_y):

        _, cluster_gen_idx = neigh_y.kneighbors(y_cond.reshape(1,-1))

        # neigh_x = NearestNeighbors(n_neighbors=cluster_size, metric='euclidean')
        # neigh_x.fit(train_x)
        # _, cluster_gen_idx = neigh_x.kneighbors(x_gen)
        cluster_gen_x = [train_x[idx.item()] for idx in cluster_gen_idx[0]]
        # cluster_gen_y = train_y[cluster_gen_idx]
        dist_list = []
        for clustered_x in cluster_gen_x:
            dist = LatticeEvaluator.compute_uc_dist_Hungary_alg(x_gen, clustered_x)
            dist_list.append(dist.mean())


        return np.min(dist_list)



    def eval_graph_validity(self):
        return self.graph_validity(self.cart_coords, self.edges, self.lattice_vectors)

    def eval_diversity(self):
        train_x =  [x.cart_coords for x in self.test_dataset]

        metrics_dict, _ = self.compute_cov(self.cart_coords, train_x, self.diversity_error_bar)
        print(metrics_dict)
        return metrics_dict['cov_recall'], metrics_dict['cov_precision']

    @staticmethod
    def compute_uc_dist_Hungary_alg(coord, gt_coord):
        dist_ij = cdist(coord, gt_coord)
        _, col_indices = linear_sum_assignment(dist_ij)
        selected_values = dist_ij[np.arange(col_indices.shape[0]), col_indices]
        return selected_values

    @staticmethod
    def compute_uc_dist_greedy_alg(coord, gt_coord):
        matrix = cdist(coord, gt_coord)
        n, m = matrix.shape
        sorted_indices = np.argsort(matrix, axis=1)

        pointers = np.zeros(n, dtype=int)

        used_indices = set()
        result = np.full(n, -1, dtype=int)

        for i in range(n):
            while pointers[i] < m:
                candidate = sorted_indices[i, pointers[i]]
                if candidate not in used_indices:
                    used_indices.add(candidate)
                    result[i] = candidate
                    break
                else:
                    pointers[i] += 1
            if pointers[i] >= m:
                break
        if i < len(result):
            result[i] = np.argmin(matrix[i])
            i += 1
        result_values = matrix[np.arange(result.shape[0]), result]
        return result_values

    @staticmethod
    def compute_cov(coords, gt_coords,
                    struc_cutoff, num_gen_strcuture=None):

        struc_pdist = np.zeros((len(coords), len(gt_coords)))
        for i in tqdm(range(len(coords)),desc='Computing Cov'):
            for j in range(len(gt_coords)):
                values = LatticeEvaluator.compute_uc_dist_Hungary_alg(coords[i], gt_coords[j])
                # values = LatticeEvaluator.compute_uc_dist_greedy_alg(coords[i], gt_coords[j])
                struc_pdist[i, j] = values.mean()

        if num_gen_strcuture is None:
            num_gen_crystals = len(coords)

        struc_recall_dist = struc_pdist.min(axis=0)
        struc_precision_dist = struc_pdist.min(axis=1)

        cov_recall = np.mean(
            struc_recall_dist <= struc_cutoff)
        cov_precision = np.sum(
            struc_precision_dist <= struc_cutoff) / num_gen_crystals

        metrics_dict = {
            'cov_recall': cov_recall,
            'cov_precision': cov_precision,
            'amsd_recall': np.mean(struc_recall_dist),
            'amsd_precision': np.mean(struc_precision_dist),
        }

        combined_dist_dict = {
            'struc_recall_dist': struc_recall_dist.tolist(),
            'struc_precision_dist': struc_precision_dist.tolist(),
        }

        return metrics_dict, combined_dist_dict


    def graph_validity(self, coords: List, edges: List, lattice_vectors: List):
        '''

        Args:
            coords:  List(np.ndarray(size=(n,3)))
            lattice_vectors: List(np.ndarray())
            edges:  List(np.ndarray((2, m)))

        Returns:
            periodicity_ratio, mean_symmetry, connectivity_ratio
        '''
        periodicity = []
        connectivity = []
        symmetry_ratio = []
        dangling_node = []
        for i in tqdm(range(len(coords)), desc='Graph validity'):
            lattice_vector = lattice_vectors[i]

            periodicity.append(
                self.is_periodic_necessary_condition(coords[i], lattice_vector.reshape(3, 3), error_bar=self.periodic_error_bar))
            connectivity.append(self.is_connected(edges[i]))
            symmetry_ratio.append(self.central_symmetry(coords[i], error_bar=self.central_symmetry_error_bar))
            dangling_node.append(self.has_dangling_node(coords[i], edges[i]))


        periodicity_ratio = np.array(periodicity).sum() / len(periodicity)
        print(f"Periodicity rate: {periodicity_ratio}")
        mean_symmetry = np.array(symmetry_ratio).mean()
        print(f"Mean Central Symmetry rate: {mean_symmetry}")
        connectivity_ratio = np.array(connectivity).sum() / len(connectivity)
        print(f"Connectivity rate: {connectivity_ratio}")
        dangling_node_ratio = 1- np.array(dangling_node).sum() / len(dangling_node)
        print(f'Dangling restriction rate: {dangling_node_ratio}')

        # print(f'Overall validity: ')
        return periodicity_ratio, mean_symmetry, connectivity_ratio, dangling_node_ratio


    @staticmethod
    def has_dangling_node(coords, edge_index):
        if edge_index is None or (edge_index == np.array(None)).any():
            return True
        if edge_index.shape[0] != 2:
            edge_index = edge_index.T

        i, j = edge_index

        degree_dict = {atom_idx: 0 for atom_idx in range(len(coords))}

        for start_node, end_node in zip(i, j):
            degree_dict[start_node] += 1
            degree_dict[end_node] += 1

        for degree in degree_dict.values():
            if degree <= 1:
                return True

        return False


    # TODO:  @Wangzhi
    @staticmethod
    def periodical_ratio():
        '''
        TODO:  @Wangzhi
        Returns:

        '''
        pass


    @staticmethod
    def is_periodic_necessary_condition(coords, lattice_vector, error_bar=1e-5):
        # input numpy array
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        if isinstance(lattice_vector, torch.Tensor):
            lattice_vector = lattice_vector.cpu().numpy()

        for d in range(3):
            find_period = False
            for i in range(len(coords)):
                new_coords_i = coords[i] + lattice_vector[d]
                dist = np.abs(new_coords_i - coords)
                if np.any(np.isclose(dist.sum(-1), 0, atol=error_bar)):
                    find_period = True
                    break
            if not find_period:
                return False
        return True

    @staticmethod
    def is_connected(edges):
        if edges is None or (edges == np.array(None)).any():
            return False
        if edges.shape[0] == 2:
            edges = edges.T
        G = nx.Graph()
        G.add_edges_from(edges)
        return nx.is_connected(G)

    @staticmethod
    def central_symmetry(coords, lattice_vector=None, use_lattice_center=False, error_bar=1e-1):
        '''
        When all nodes are symmetry nodes: 𝐶𝑒𝑛𝑡𝑟𝑦 𝑠𝑦𝑚𝑚𝑒𝑡𝑟𝑦   = 1
        When no nodes are symmetry: 𝐶𝑒𝑛𝑡𝑟𝑦 𝑠𝑦𝑚𝑚𝑒𝑡𝑟𝑦   = 0
        More #Symmetry nodes, larger 𝐶𝑒𝑛𝑡𝑟𝑦 𝑠𝑦𝑚𝑚𝑒𝑡𝑟𝑦  value.
        Less error of symmetry node, larger 𝐶𝑒𝑛𝑡𝑟𝑦 𝑠𝑦𝑚𝑚𝑒𝑡𝑟𝑦  value
        None symmetry node position won’t influence the 𝐶𝑒𝑛𝑡𝑟𝑦 𝑠𝑦𝑚𝑚𝑒𝑡𝑟𝑦 value.

        Args:
            coords:
            lattice_vector:
            use_lattice_center:
            error_bar:

        Returns:

        '''
        # if coords1 is symmetry to coords2: coords1 - central == -(coords2 - central)
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        if use_lattice_center:
            assert lattice_vector is not None
            center = np.array([[0.,0.,0.]])
            center = np.einsum('bi,bij->bj', center.float(), lattice_vector.unsqueeze(0).float())
        else:
            center = np.array([(coords[:,0].max() + coords[:,0].min()) / 2,
                               (coords[:,1].max() + coords[:,1].min()) / 2,
                               (coords[:,2].max() + coords[:,2].min()) / 2 ])

        dist = coords - center
        dist2 = np.expand_dims(dist, axis=1)
        new_dist = dist + dist2
        descarts_central_dist = np.square(new_dist).sum(axis=-1)**(0.5)
        symmetry_num_per_node = (np.isclose(descarts_central_dist, 0,atol=error_bar)).sum(axis=-1)
        is_symmetry_per_node = symmetry_num_per_node > 0
        symmetry_node_num = is_symmetry_per_node.sum() + 1e-6
        symmetry_node_rate = is_symmetry_per_node.sum() / coords.shape[0]  # Sn

        max_error = max(((dist)**(2)).sum(axis=-1)**(0.5))
        s_error_i = ((new_dist**2).sum(axis=-1)**(0.5)).min(axis=-1)
        s_error_i_ratio = (max_error - s_error_i) / max_error

        

        central_symmetry_rate = symmetry_node_rate * (1 / symmetry_node_num) * (is_symmetry_per_node * s_error_i_ratio).sum()
        return central_symmetry_rate






if __name__ == '__main__':
    ## Example for evaluating generation task
    '''
    Saving lattices:
            np.savez(lattice_name,
                atom_types=gen_atom_types_list[i],
                lengths=gen_lengths_list[i],
                angles=gen_angles_list[i],
                frac_coords=gen_frac_coords_list[i],
                edge_index=edge_index_list[i],
                prop_list=prop_list[i]
                )
    - `cart_coords`: None or cartesian coordinates of each atom, shape `(num_evals, N, 3)`
    - `frac_coords`: fractional coordinates of each atom, shape `(num_evals, N, 3)`
    - `atom_types`: atomic number of each atom, shape `(num_evals, N)`
    - `lengths`: the lengths of the lattice, shape `(num_evals, M, 3)`
    - `angles`: the angles of the lattice, shape `(num_evals, M, 3)`
    - `num_atoms`: the number of atoms in each material, shape `(num_evals, M)`


    The following codes will print:
    {'cov_recall': 0.0, 'cov_precision': 0.0, 'amsd_recall': 10.602191118204289, 'amsd_precision': 13.421280170913064}
    Periodicity rate: 0.0
    Mean Central Symmetry rate: 0.5374038704332712
    Connectivity rate: 1.0
    Dangling rate: 0.0
    '''
    dataset = LatticeModulus('D:\\项目\\Material design\\code_data\\data\\LatticeModulus',file_name='data')
    split_dict = dataset.get_idx_split(len(dataset), 8000, 2000, seed=42)
    test_data = dataset[split_dict['test']]
    evaluator = LatticeEvaluator(test_datset=test_data[:1000], eval_file_path='D:\\Workspace\\PhD_workspace\\metabench-proj\\Metamaterial-Benchmark\\gen_results\\edm\\save_results')

    effectiveness = evaluator.eval_condition_effectiveness()


    evaluator.evaluate_all_uncondition_generation()
    # cart_coords: List[np.ndarray] = [d.cart_coords.numpy() for d in dataset]
    # frac_coords: List[np.ndarray] = [d.frac_coords.numpy() for d in dataset]
    # node_types: List[np.ndarray] = [d.node_type.numpy() for d in dataset]
    # edges: List[np.ndarray] = [d.edge_index.numpy() for d in dataset]
    # lattice_vectors: List[np.ndarray] = [d.vector.numpy() for d in dataset]
    #
    # evaluator = LatticeEvaluator(
    #     None,None,
    #     cart_coords,
    #     frac_coords,
    #     node_types,
    #     edges,
    #     lattice_vectors)
    # evaluator.evaluate_all_uncondition_generation()
    # Example for evaluating conditional generation task
    # y_cond = torch.randn((1, 21))  # condition
    # x_gen = torch.randn((15, 3))   # generated lattice conditioned on y_cond

