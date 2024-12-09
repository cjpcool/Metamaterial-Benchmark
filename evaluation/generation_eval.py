import os
import sys
import warnings

from scipy.spatial.distance import cdist

from datasets.dataset_truss import LatticeStiffness

sys.path.append('../')
from typing import List

import torch
import numpy as np

import networkx as nx
from jedi.api import file_name
from scipy.spatial import distance
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from datasets.baseDataset import LatticeTruss
from eval_utils import lattice_params_to_matrix, frac_to_cart_coords
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.utils import shuffle


def pad_to_max_length(fps_list,max_length, padding_value=0):
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
                 data_size_for_eval: int = None,
                 central_symmetry_error_bar=1e-1,
                 periodic_error_bar=1e-5,
                 diversity_error_bar=0.2):

        super().__init__(
            cart_coords,
            frac_coords,
            node_types,
            edges,
            lattice_vectors
        )

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


    def evaluate_all_uncondition_generation(self):
        cov_r, cov_p = self.eval_diversity()
        periodicity_ratio, mean_symmetry, connectivity_ratio, dangling_node_ratio = self.eval_graph_validity()


    def eval_condition_effectiveness(self, y_cond, x_gen):
        '''

        :param y_cond: conditioned property, shape=(1,prop_dim)
        :param x_gen: generated coordinates, shape=(node_num, 3)
        :return:
        '''
        node_num = x_gen.shape[0]
        idx_node_num = [i for i in range(len(self.test_dataset)) if self.test_dataset[i].num_nodes == node_num]
        if self.data_size_for_eval > len(idx_node_num):
            warnings.warn(f'Node number {node_num} of evaluated lattice is smaller than data_size_for_eval, setting eval size to {len(idx_node_num)}')
            right = len(idx_node_num)
        else:
            right = self.data_size_for_eval

        selected_idx = shuffle(idx_node_num)[:right]
        selected_dataset =  self.test_dataset.copy(selected_idx)
        cluster_size = min(self.cluster_size, len(selected_idx))

        dist = self.condition_effectiveness(y_cond, x_gen, selected_dataset, node_num, cluster_size)
        print(dist)
        return dist

    def eval_graph_validity(self):
        return self.graph_validity(self.cart_coords, self.edges, self.lattice_vectors)

    def eval_diversity(self):
        train_x =  [x.cart_coords for x in self.test_dataset]

        metrics_dict, _ = self.compute_cov(self.cart_coords, train_x, self.diversity_error_bar)
        print(metrics_dict)
        return metrics_dict['cov_recall'], metrics_dict['cov_precision']


    @staticmethod
    def compute_cov(coords, gt_coords,
                    struc_cutoff, num_gen_strcuture=None):
        struc_fps = [c for c in coords]
        gt_struc_fps = [c for c in gt_coords]

        # Use number of crystal before filtering to compute COV
        if num_gen_strcuture is None:
            num_gen_crystals = len(struc_fps)

        max_length = max(max(fp.shape[0] for fp in gt_struc_fps),max(fp.shape[0] for fp in struc_fps))
        struc_fps = pad_to_max_length(struc_fps, max_length=max_length)
        gt_struc_fps = pad_to_max_length(gt_struc_fps,max_length)

        struc_pdist = cdist(struc_fps.reshape(struc_fps.shape[0], -1), gt_struc_fps.reshape(gt_struc_fps.shape[0], -1))

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

    def __read_eval_data(self, eval_file_path):
        file_names = os.listdir(eval_file_path)
        for file_name in file_names:
            full_path = os.path.join(eval_file_path, file_name)
            lattice_npz = np.load(full_path)
            frac_coord = lattice_npz['frac_coords']
            self.frac_coords.append(frac_coord)

            lattice_lengths = lattice_npz['lengths']
            lattice_angles = lattice_npz['angles']

            atom_types = lattice_npz['atom_types']
            # Removed unused variable declaration
            edge_index = lattice_npz['edge_index']
            self.node_types.append(atom_types)
            self.edges.append(edge_index)

            try:
                lattice_vector = lattice_npz['vector']
            except:
                lattice_vector = lattice_params_to_matrix(lattice_lengths[0],lattice_lengths[1],lattice_lengths[2],
                                                  lattice_angles[0], lattice_angles[1], lattice_angles[2])
            self.lattice_vectors.append(lattice_vector)
            num_atoms = len(frac_coord)

            try:
                cart_coord = lattice_npz['cart_coords']
            except:
                cart_coord = frac_to_cart_coords(frac_coord,
                                    lattice_vector,
                                    num_atoms)
            self.cart_coords.append(cart_coord)


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
        for i in range(len(coords)):
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
        dangling_node_ratio = np.array(dangling_node).sum() / len(dangling_node)
        print(f'Dangling rate: {dangling_node_ratio}')

        return periodicity_ratio, mean_symmetry, connectivity_ratio, dangling_node_ratio


    @staticmethod
    def has_dangling_node(coords, edge_index):
        if edge_index.shape[0] != 2:
            edge_index = edge_index.T

        i, j = edge_index

        degree_dict = {atom_idx: 0 for atom_idx in range(len(coords))}

        for start_node, end_node in zip(i, j):
            degree_dict[start_node] += 1
            degree_dict[end_node] += 1

        for degree in degree_dict.values():
            if degree == 1:
                return True

        return False

    @staticmethod
    def condition_effectiveness(y_cond, x_gen, test_data, node_num, cluster_size=100):
        # only support for same node numbers.
        data_num = len(test_data)
        train_x = test_data.data.cart_coords.view(data_num, -1)
        train_y = test_data.data.y.view(data_num, -1)

        # train_x = train_x.cpu().numpy()
        # train_y = train_y.cpu().numpy()
        # x_cond = x_cond.reshape(1, -1)
        x_gen = x_gen.reshape(y_cond.shape[0], -1)

        neigh_x = NearestNeighbors(n_neighbors=cluster_size, metric='euclidean')
        neigh_x.fit(train_x)
        _, cluster_gen_idx = neigh_x.kneighbors(x_gen)
        cluster_gen_x = train_x[cluster_gen_idx]
        cluster_gen_y = train_y[cluster_gen_idx]

        # neigh_y = NearestNeighbors(n_neighbors=cluster_size, metric='cosine')
        # neigh_y.fit(train_y)
        # _, cluster_cond_idx = neigh_y.kneighbors(y_cond)
        # cluster_cond_x = train_x[cluster_cond_idx]

        # label_cluster_cond = np.zeros((cluster_size,))
        # label_cluster_gen = np.ones((cluster_size,))
        #
        # labels = np.concatenate((label_cluster_cond, label_cluster_gen), axis=0)
        # x = np.concatenate((cluster_cond_x, cluster_gen_x), axis=0)
        # kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
        # res = kmeans.fit_predict(x)
        # nmi = nmi_score(labels, res, average_method='arithmetic')

        distance,_ = np.sqrt(((cluster_gen_y - y_cond)**2).sum(axis=-1)).min(axis=1)

        return distance.item()



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
    The following codes will print:
    {'cov_recall': 0.0, 'cov_precision': 0.0, 'amsd_recall': 10.602191118204289, 'amsd_precision': 13.421280170913064}
    Periodicity rate: 0.0
    Mean Central Symmetry rate: 0.5374038704332712
    Connectivity rate: 1.0
    Dangling rate: 0.0
    '''
    dataset = LatticeStiffness('D:\项目\Material design\code_data\data\LatticeStiffness')
    evaluator = LatticeEvaluator(test_datset=dataset, eval_file_path='D:\\Workspace\\PhD_workspace\\MetaMatGen\\generated_mat\\lattices\\lattices')
    evaluator.evaluate_all_uncondition_generation()

    # Example for evaluating conditional generation task
    y_cond = torch.randn((1, 21))  # condition
    x_gen = torch.randn((15, 3))   # generated lattice conditioned on y_cond
    effectiveness = evaluator.eval_condition_effectiveness(y_cond, x_gen)
