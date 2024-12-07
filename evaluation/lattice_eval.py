import os
import sys
sys.path.append('../')
print(sys.path)
from typing import List

import torch
import numpy as np

import networkx as nx
from jedi.api import file_name
from scipy.spatial import distance
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from utils import find_lattice_vectors
from datasets.dataset_truss import LatticeTruss
from utils import lattice_params_to_matrix, frac_to_cart_coords
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.utils import shuffle



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
    '''
    Evaluate generated lattice:
        Graph Level validity:
            1. Central symmetry ratio:
            2. Periodic boundary conditions test
            3. Connectivity test
            4. Periodic ratio: TODO: wangzhi PBC ratio
        Condition guide effectiveness:
            1.

        lattice validity:
            TODO

    Statistic evaluation on mass generated data.

    Evaluate reconstructed graph:
        Edge Level:
            TODO:
        Node Level:
            TODO:

    '''
    def __init__(self,
                 test_datset: LatticeTruss=None,
                 eval_file_path: str=None,
                 cart_coords: List[np.ndarray] = None,
                 frac_coords: List[np.ndarray] = None,
                 node_types: List[np.ndarray] = None,
                 edges: List[np.ndarray] = None,
                 lattice_vectors: List[np.ndarray] = None,
                 cluster_size: int = 100,
                 data_size_for_eval: int = None,
                 central_symmetry_error_bar=1e-1,
                 periodic_error_bar=1e-5):
        super().__init__(
            cart_coords,
            frac_coords,
            node_types,
            edges,
            lattice_vectors
        )

        self.central_symmetry_error_bar = central_symmetry_error_bar
        self.periodic_error_bar = periodic_error_bar

        if eval_file_path is not None:
            self.__read_eval_data(eval_file_path)

        self.test_dataset = test_datset
        self.cluster_size = cluster_size
        if data_size_for_eval is not None:
            self.data_size_for_eval = data_size_for_eval
        else:
            self.data_size_for_eval = len(test_datset)
        assert self.cluster_size > self.data_size_for_eval, 'data_size_for_eval < cluster_size'


    def eval_condition_effectiveness(self, y_cond, x_gen):
        node_num = x_gen.shape[0]
        idx_node_num = np.array([i for  i in range(len(self.test_dataset)) if self.test_dataset[i].num_nodes == node_num])
        if self.data_size_for_eval > len(idx_node_num):
            print(f'Node number f{node_num} of evaluated lattice is smaller than hyperparameters, setting eval size to f{len(idx_node_num)}')
            right = len(idx_node_num)
        else:
            right = self.data_size_for_eval
        selected_idx = shuffle(idx_node_num)[:right]
        selected_dataset =  self.test_dataset[selected_idx]
        return self.condition_effectiveness(y_cond, x_gen, selected_dataset, node_num, self.cluster_size)

    def eval_graph_validity(self):
        return self.graph_validity(self.cart_coords, self.edges, self.lattice_vectors)



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
        for i in range(len(coords)):
            lattice_vector = lattice_vectors[i]

            periodicity.append(
                self.is_periodic_necessary_condition(coords[i], lattice_vector.reshape(3, 3), error_bar=self.periodic_error_bar))
            connectivity.append(self.is_connected(edges[i]))
            symmetry_ratio.append(self.central_symmetry(coords[i], error_bar=self.central_symmetry_error_bar))


        periodicity_ratio = np.array(periodicity).sum() / len(periodicity)
        print(f"Periodicity rate: {periodicity_ratio}")
        mean_symmetry = np.array(symmetry_ratio).mean()
        print(f"Mean Central Symmetry rate: {mean_symmetry}")
        connectivity_ratio = np.array(connectivity).sum() / len(connectivity)
        print(f"Connectivity rate: {connectivity_ratio}")

        return periodicity_ratio, mean_symmetry, connectivity_ratio



    @staticmethod
    def condition_effectiveness(y_cond, x_gen, test_data, node_num, cluster_size=100):
        # only suport for same node numbers.
        train_y, train_x = test_data.y, test_data.pos
        if isinstance(y_cond, torch.Tensor):
            y_cond = y_cond.cpu().numpy()
        # if isinstance(x_cond, torch.Tensor):
        #     x_cond = x_cond.cpu().numpy()
        if isinstance(x_gen, torch.Tensor):
            x_gen = x_gen.cpu().numpy()
        if isinstance(train_y, torch.Tensor):
            train_y = train_y.cpu().numpy()
        if isinstance(train_x, torch.Tensor):
            train_x = train_x.cpu().numpy()

        data_num = train_y.shape[0] // node_num
        train_y = train_y.reshape[data_num, -1]
        # x_cond = x_cond.reshape(1, -1)
        x_gen = x_gen.reshape(1, -1)

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

        distance = np.sqrt(((cluster_gen_y - y_cond)**2).sum(axis=1)).mean()

        return distance



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





def evaluate_lattice_in_path(path, error_bar=0.2, refind_lattice_vector=False):
    import os
    file_names = os.listdir(path)
    periodicity = []
    connectivity = []
    symmetry_ratio = []

    for file_name in file_names:
        full_path = os.path.join(path, file_name)
        lattice_npz = np.load(full_path)
        frac_coords = lattice_npz['frac_coords']
        lattice_lengths = lattice_npz['lengths']
        lattice_angles = lattice_npz['angles']
        atom_types = lattice_npz['atom_types']
        edge_index = lattice_npz['edge_index']
        if refind_lattice_vector:
            lattice_vector = find_lattice_vectors(frac_coords)
        else:
            try:
                lattice_vector = lattice_npz['vector']
            except:
                lattice_vector = lattice_params_to_matrix(lattice_lengths[0],lattice_lengths[1],lattice_lengths[2],
                                                  lattice_angles[0], lattice_angles[1], lattice_angles[2])

        periodicity.append(LatticeEvaluator.is_periodic_necessary_condition(frac_coords, lattice_vector.reshape(3,3), error_bar=error_bar))
        connectivity.append(LatticeEvaluator.is_connected(edge_index))
        symmetry_ratio.append(LatticeEvaluator.central_symmetry(frac_coords, error_bar = error_bar))

    print(periodicity)
    print(connectivity)
    print(symmetry_ratio)



    periodicity_ratio = np.array(periodicity).sum() / len(periodicity)
    print(f"Periodicity rate: {periodicity_ratio}")
    mean_symmetry = np.array(symmetry_ratio).mean()
    print(f"Mean Central Symmetry: {mean_symmetry}")
    # print(f"Valid Central Symmetry rate: {(np.array(symmetry_ratio)>0).sum() / len(symmetry_ratio)}")

    connectivity_ratio = np.array(connectivity).sum() / len(connectivity)
    print(f"Connectivity rate: {connectivity_ratio}")











if __name__ == '__main__':
    nodes = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
         [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    edges = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]).T
    lattice_lengths = np.array([1.0, 1.0, 1.0])  #
    lattice_angles = np.array([90, 90, 90])  #


    # lattice_vectors = find_lattice_vectors(nodes)
    # print("Lattice Vectors:\n", lattice_vectors)
    # periodicity = LatticeEvaluator.central_symmetry(nodes, error_bar=0.1)
    # print("Periodicity:", periodicity)


    evaluate_lattice_in_path('D:\\Workspace\\PhD_workspace\\MetaMatGen\\generated_mat\\lattices\\lattices')
    #
    #
    # lattice_vector = lattice_params_to_matrix(lattice_lengths[0],lattice_lengths[1],lattice_lengths[2],
    #                                           lattice_angles[0], lattice_angles[1], lattice_angles[2])
    #
    #
    # periodicity = is_periodic_necessary_condition(nodes, lattice_vector)
    # periodicity = is_periodic(nodes, lattice_lengths)
    # print(f"Periodicity: {periodicity}")
    #
    # connectivity = is_connected(edges)
    # print(f"Connectivity: {connectivity}")
    #
    # symmetry = central_symmetry(nodes)
    # print(f"Symmetry: {symmetry}")