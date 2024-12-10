import os
import sys
import warnings
import copy
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, cg
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spilu

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

        self.property_calculator = Property_Calculator()

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
    def periodical_degree(self, x_gen, heuristic_attempts=500):
        # this method suppose x_gen to be batch_size * (3node_num) or batch_size * node_num * 3
        # return a periodicity degree measure, between 0, 1, with 0 being the best, 1 the worst
        x_gen1 = copy.deepcopy(x_gen).reshape((x_gen.shape[0],-1,3)).astype(np.float64)
        x_gen1 -= np.mean(x_gen1,axis=1,keepdims=True)
        us = np.zeros((x_gen1.shape[0],3,3))
        for i in range(x_gen1.shape[0]):
            u = np.random.random((3,3))
            u = self.__cal_axis(u[0],u[1])
            temp_dist = self.__cal_largest_poker(x_gen1[i],u)
            for _ in range(3):
                u1 = np.random.random((3,3))
                u1 = self.__cal_axis(u[0],u[1])
                if self.__cal_largest_poker(x_gen1[i],u1) < temp_dist:
                    temp_dist = self.__cal_largest_poker(x_gen1[i],u1)
                    u = u1
            for j in range(heuristic_attempts):
                ini_u = copy.deepcopy(u)
                dist0 = self.__cal_largest_poker(x_gen1[i],np.array(u))
                u = self.__shift_axis(u[0],u[1],amplitude=np.sqrt(j)*0.002)
                dist1 = self.__cal_largest_poker(x_gen1[i],np.array(u))
                if dist1 > dist0:
                    u = ini_u
                else:
                    u -= 0.2*(ini_u - u)
            us[i] += u
        dev = self.__cal_deviation(us,x_gen1)
        return dev
    
    def __cal_axis(self, u0, u1, u2=None):
        u0 = u0 / np.linalg.norm(u0,axis=-1,keepdims=True)
        u1 = u1 - np.sum(u0*u1,axis=-1)*u0
        u1 = u1 / np.linalg.norm(u1,axis=-1,keepdims=True)
        u2 = np.cross(u0,u1,axis=-1)
        u2 = u2 / np.linalg.norm(u2,axis=-1,keepdims=True)
        return np.array([u0, u1, u2])
    
    def __shift_axis(self, u0, u1, u2=None, amplitude=0.01):
        axis_0 = np.random.rand() > 0.5
        which_dim = np.random.choice([0, 1, 2])
        increase = np.random.rand() > 0.5
        if increase:
            if axis_0:
                #print(u0.shape)
                u0[which_dim] += amplitude
            else:
                u1[which_dim] += amplitude
        else:
            if axis_0:
                #print(u0.shape)
                u0[which_dim] -= amplitude
            else:
                u1[which_dim] -= amplitude
        return self.__cal_axis(u0, u1)

    def __cal_largest_poker(self, x_gen_one, u):
        dists = np.abs(np.sum(x_gen_one[:, np.newaxis, :] * u[np.newaxis, :, :],axis=-1))
        max_dist = np.max(dists)
        return max_dist
        
    def __cal_deviation(self, us, x_gen, shell_threshold=0.9):
        dists = np.sum(x_gen[:, :, np.newaxis, :] * us[:, np.newaxis, :, :],axis=-1)
        dev = np.zeros(x_gen.shape[0])
        for i in range(x_gen.shape[0]):
            temp_dev = []
            half_size = np.max(np.abs(dists[i]))
            for j in range(dists.shape[1]):
                #not a shell node
                if np.max(np.abs(dists[i,j,:])) < shell_threshold * half_size:
                    continue
                dist = 4*half_size
                for k in range(dists.shape[1]):
                    if j == k: continue
                    temp_vec = np.abs(dists[i,j] - dists[i,k])
                    temp_vec[np.argmax(temp_vec)] = np.abs(1 - temp_vec[np.argmax(temp_vec)])
                    if np.linalg.norm(temp_vec) < dist:
                        dist = np.linalg.norm(temp_vec)
                temp_dev.append(dist)
            dev[i] = np.mean(temp_dev)
        return dev
    
    def cal_property(self,coords,edges,radius=0.1):
        return self.property_calculator.calculate_properties(coords,edges,radius)


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

class Property_Calculator:
    def calculate_relative_density(self, coords, edges, radius=0.1):
        lengths = np.zeros(len(edges))
        for i, edge in enumerate(edges):
            start = coords[edge[0]]  # -1 since edges are 1-indexed
            end = coords[edge[1]]
            lengths[i] = np.linalg.norm(end - start)
        length = np.sum(lengths)
        volume = length * np.pi * radius**2
        density = volume / 1

        return density

    def calculate_properties(self,coords, edges, radius=0.1, resolution_voxel=40):
        voxel, Density = self.generate_voxel(resolution_voxel,(coords, edges), radius)
        CH = self.homo3D(1,1,1,0.5769,0.3846, voxel)
        properties = self.from_C_to_properties(CH)
        density = self.calculate_relative_density(coords, edges, radius)
        properties['density'] = density
        properties['Density'] = Density

        return properties

    # def calculate_properties(coords, edges):
    #     voxel, Density = generate_voxel(10,(coords, edges),0.1)
    #     CH = homo3D(1,1,1,0.5769,0.3846, voxel)
    #     properties = from_C_to_properties(CH)
    #     return properties

    def generate_voxel(self,n, lattice, radius):
        """
        Generate a voxel grid and calculate the relative density.
        
        Parameters:
            n (int): Number of voxels along each axis.
            address (str): File location of the wireframe.
            radius (float): Radius for determining active voxels.
            
        Returns:
            tuple: Voxel grid (3D numpy array) and density (float).
        """
        size = 1.0 / n               # initial size of voxels
        voxel = np.zeros((n, n, n))  # initial grid with zeros

        # Generate a list of centers of voxel
        voxel_c = np.zeros((n**3, 6))
        p = 0                        # p count the number of all voxels
        for i in range(1, n + 1):    # i for z axis
            for j in range(1, n + 1):# j for y axis
                for k in range(1, n + 1): # k for x axis
                    p += 1
                    voxel_c[p-1, 0:3] = [k, j, i]  # save index along x,y,z axis
                    # save coordinate along x,y,z axis
                    voxel_c[p-1, 3:6] = [(k-0.5)*size, (j-0.5)*size, (i-0.5)*size]

        # Get the voxel close to the strut within a certain distance
        node, strut = lattice # get the information of strut
        for i in range(len(voxel_c)):      # for each voxel, decide if it is active
            center = voxel_c[i, 3:6]       # voxel center position
            for j in range(len(strut)):    # for each strut, get the distance to the voxel
                # start_n = node[strut[j, 0] - 1, :]  # start node coordinate
                start_n = node[strut[j, 0], :]  # start node coordinate
                end_n = node[strut[j, 1], :]    # end node coordinate

                # determine if alpha and beta are acute angles
                alpha = np.degrees(np.arccos(np.dot((center - start_n), (end_n - start_n)) / 
                                            (np.linalg.norm(center - start_n) * np.linalg.norm(end_n - start_n))))
                beta = np.degrees(np.arccos(np.dot((center - end_n), (start_n - end_n)) / 
                                        (np.linalg.norm(center - end_n) * np.linalg.norm(start_n - end_n))))

                if alpha < 90 and beta < 90:  # if not acute angle, distance to line
                    distance = np.linalg.norm(np.cross(end_n - start_n, center - start_n)) / np.linalg.norm(end_n - start_n)
                else:                         # if it is acute angle, distance to node
                    distance = min(np.linalg.norm(center - start_n), np.linalg.norm(center - end_n))
                
                if distance <= radius:        # if distance less than radius, activate it
                    voxel[int(voxel_c[i, 0]) - 1, int(voxel_c[i, 1]) - 1, int(voxel_c[i, 2]) - 1] = 1
                    break  # move to the next voxel

        density = np.sum(voxel) / n**3  # calculate the relative density
        
        return voxel, density



    def homo3D(self,lx, ly, lz, lambda_, mu, voxel):
        """
        Calculate the effective stiff matrix of lattice structure from the voxel data.
        
        Parameters:
            lx, ly, lz (float): unit cell size.
            lambda_, mu (float): Material properties
            voxel (NxNxN array): structure of the lattice
        Returns:
            CH (6x6 array): the effective stiff matrix Cijkl.
        """
        # Initialize
        nelx, nely, nelz = voxel.shape
        dx = lx / nelx
        dy = ly / nely
        dz = lz / nelz
        nel = nelx * nely * nelz

        # Compute element stiffness matrices
        keLambda, keMu, feLambda, feMu = self.hexahedron(dx/2, dy/2, dz/2)

        # Node numbers and element degrees of freedom for full (not periodic) mesh
        nodenrs = np.arange(1, (1 + nelx) * (1 + nely) * (1 + nelz) + 1).reshape((1 + nelx, 1 + nely, 1 + nelz))
        edofVec = (3 * nodenrs[:-1, :-1, :-1] + 1).flatten()
        # addx = np.append([0, 1, 2], [3 * nelx + np.array([3, 4, 5, 0, 1, 2])])
        # addx = np.append(addx, [-3, -2, -1])
        addx = np.append([0, 1, 2], [3 * nelx + np.array([3, 4, 5, 0, 1, 2])])
        addx = np.append(addx, [-3, -2, -1])
        addxy = 3 * (nely + 1) * (nelx + 1) + addx
        edof = np.tile(edofVec[:, np.newaxis], (1, 24)) + np.tile(np.concatenate([addx, addxy]), (nel, 1))

        # Impose periodic boundary conditions
        nn = (nelx + 1) * (nely + 1) * (nelz + 1)  # Total number of nodes
        nnP = nelx * nely * nelz  # Total number of unique nodes
        nnPArray = np.arange(1, nnP + 1).reshape(nelx, nely, nelz)
        nnPArray = np.pad(nnPArray, ((0, 1), (0, 1), (0, 1)), mode='wrap')
        dofVector = np.zeros(3 * nn, dtype=int)
        dofVector[0::3] = 3 * nnPArray.flatten() - 2
        dofVector[1::3] = 3 * nnPArray.flatten() - 1
        dofVector[2::3] = 3 * nnPArray.flatten()
        #edof = dofVector[edof.flatten()].reshape(edof.shape)
        edof = edof-1
        edof = dofVector[edof]
        
        ndof = 3 * nnP

        # ASSEMBLE GLOBAL STIFFNESS MATRIX AND LOAD VECTORS
        # Indexing vectors
        iK = np.kron(edof, np.ones((24, 1))).T
        jK = np.kron(edof, np.ones((1, 24))).T
        # Material properties assigned to voxels with materials
        lambda_ = lambda_ * (voxel == 1)
        mu = mu * (voxel == 1)
        # The corresponding stiffness matrix entries
        sK = np.outer(keLambda.flatten('F'), lambda_.flatten('F')) + np.outer(keMu.flatten('F'), mu.flatten('F'))
        K = csr_matrix((sK.flatten('F'), (iK.flatten('F')-1, jK.flatten('F')-1)), shape=(ndof, ndof))
        K = 0.5 * (K + K.T)

        # Assembly three load cases corresponding to the three strain cases
        iF = np.tile(edof.T, (6, 1))
        jF = np.vstack([np.ones((24, nel)), 2 * np.ones((24, nel)), 3 * np.ones((24, nel)),
                        4 * np.ones((24, nel)), 5 * np.ones((24, nel)), 6 * np.ones((24, nel))])
        sF = np.outer(feLambda.flatten('F'), lambda_.flatten('F')) + np.outer(feMu.flatten('F'), mu.flatten('F'))
        F = csr_matrix((sF.flatten('F'), (iF.flatten('F')-1, jF.flatten('F')-1)), shape=(ndof, 6))

        # SOLUTION
        # solve by PCG method, remember to constrain one node
        activedofs = edof[voxel.flatten() == 1, :]
        activedofs = np.sort(np.unique(activedofs))
    # activedofs = activedofs-1
        X = np.zeros((ndof, 6))
        #L = splu(K[activedofs[3:], :][:, activedofs[3:]])
        
    # for i in range(6):
    #     X[activedofs[3:], i] = L.solve(F[activedofs[3:], i])
        K_act = K[activedofs[3:]-1, :][:, activedofs[3:]-1]
        K_act = csc_matrix(K_act)  # Convert matrix to CSC format
        epsilon = 1e-6
        K_act = K_act + epsilon * eye(K_act.shape[0])
        M = LinearOperator(K_act.shape, spilu(K_act).solve)
        # Ensure b is a proper 1D vector
        for i in range(6):
            b = F[activedofs[3:]-1, i].toarray().flatten()  # Convert to 1D array if it's a sparse matrix
            X[activedofs[3:]-1, i], _ = cg(K_act, b, maxiter=300, M=M)


        # HOMOGENIZATION
        # The displacement vectors corresponding to the unit strain cases
        X0 = np.zeros((nel, 24, 6))
        # The element displacements for the six unit strains
        X0_e = np.zeros((24, 6))
        ke = keMu + keLambda  # Here the exact ratio does not matter, because
        fe = feMu + feLambda  # it is reflected in the load vector
        X0_e[np.array([3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]), :] = \
            np.linalg.solve(ke[np.array([3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]), 
                                    :][:, np.array([3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])], 
                            fe[np.array([3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]), :])
        
        for i in range(6):
            X0[:, :, i] = np.kron(X0_e[:, i].T[np.newaxis, :], np.ones((nel, 1)))

        CH = np.zeros((6, 6))
        volume = lx * ly * lz
        for i in range(6):
            for j in range(6):
                sum_L = np.dot((X0[:, :, i] - X.flatten('F')[edof.flatten('F')-1 + (i) * ndof].reshape(nnP, 24, order='F')), keLambda) * \
                        (X0[:, :, j] - X.flatten('F')[edof.flatten('F')-1 + (j) * ndof].reshape(nnP, 24, order='F'))
                sum_M = np.dot((X0[:, :, i] - X.flatten('F')[edof.flatten('F')-1 + (i) * ndof].reshape(nnP, 24, order='F')), keMu) * \
                        (X0[:, :, j] - X.flatten('F')[edof.flatten('F')-1 + (j)* ndof].reshape(nnP, 24, order='F'))
                sum_L = np.reshape(np.sum(sum_L, axis=1), (nelx, nely, nelz))
                sum_M = np.reshape(np.sum(sum_M, axis=1), (nelx, nely, nelz))
                CH[i, j] = np.sum(lambda_ * sum_L + mu * sum_M)
        CH = 1 / volume * CH
        return CH


    def hexahedron(self, a, b, c):
        # Constitutive matrix contributions
        CMu = np.diag([2, 2, 2, 1, 1, 1])
        CLambda = np.zeros((6, 6))
        CLambda[0:3, 0:3] = 1
        
        # Three Gauss points in both directions
        xx = [-np.sqrt(3/5), 0, np.sqrt(3/5)]
        yy = xx
        zz = xx
        ww = [5/9, 8/9, 5/9]
        
        # Initialize
        keLambda = np.zeros((24, 24))
        keMu = np.zeros((24, 24))
        feLambda = np.zeros((24, 6))
        feMu = np.zeros((24, 6))
        
        for ii in range(len(xx)):
            for jj in range(len(yy)):
                for kk in range(len(zz)):
                    # Integration point
                    x = xx[ii]
                    y = yy[jj]
                    z = zz[kk]
                    
                    # Stress strain displacement matrix
                    qx = np.array([-((y-1)*(z-1))/8, ((y-1)*(z-1))/8, -((y+1)*(z-1))/8,
                                ((y+1)*(z-1))/8, ((y-1)*(z+1))/8, -((y-1)*(z+1))/8,
                                ((y+1)*(z+1))/8, -((y+1)*(z+1))/8])
                    
                    qy = np.array([-((x-1)*(z-1))/8, ((x+1)*(z-1))/8, -((x+1)*(z-1))/8,
                                ((x-1)*(z-1))/8, ((x-1)*(z+1))/8, -((x+1)*(z+1))/8,
                                ((x+1)*(z+1))/8, -((x-1)*(z+1))/8])
                    
                    qz = np.array([-((x-1)*(y-1))/8, ((x+1)*(y-1))/8, -((x+1)*(y+1))/8,
                                ((x-1)*(y+1))/8, ((x-1)*(y-1))/8, -((x+1)*(y-1))/8,
                                ((x+1)*(y+1))/8, -((x-1)*(y+1))/8])
                    
                    # Jacobian
                    J = np.array([qx, qy, qz]) @ np.array([[-a, a, a, -a, -a, a, a, -a],
                                                        [-b, -b, b, b, -b, -b, b, b],
                                                        [-c, -c, -c, -c, c, c, c, c]]).T
                    qxyz = np.linalg.inv(J) @ np.array([qx, qy, qz])
                    
                    B_e = np.zeros((6, 3, 8))
                    for i_B in range(8):
                        B_e[:, :, i_B] = np.array([
                            [qxyz[0, i_B], 0, 0],
                            [0, qxyz[1, i_B], 0],
                            [0, 0, qxyz[2, i_B]],
                            [qxyz[1, i_B], qxyz[0, i_B], 0],
                            [0, qxyz[2, i_B], qxyz[1, i_B]],
                            [qxyz[2, i_B], 0, qxyz[0, i_B]]
                        ])
                    
                    B = np.hstack([B_e[:, :, i] for i in range(8)])
                    
                    # Weight factor at this point
                    weight = np.linalg.det(J) * ww[ii] * ww[jj] * ww[kk]
                    
                    # Element matrices
                    keLambda += weight * B.T @ CLambda @ B
                    keMu += weight * B.T @ CMu @ B
                    
                    # Element loads
                    feLambda += weight * B.T @ CLambda
                    feMu += weight * B.T @ CMu
        
        return keLambda, keMu, feLambda, feMu


    def from_C_to_properties(self, C):
        # Add a small value to diagonal elements to ensure matrix is not singular
        # Check if C is singular or nearly singular
        # Check if input is a 2D array
        if not isinstance(C, np.ndarray) or C.ndim != 2:
            raise ValueError("Input C must be a 2D numpy array")
            
        # Check if matrix is square
        if C.shape[0] != C.shape[1]:
            raise ValueError("Input C must be a square matrix")
        if np.linalg.matrix_rank(C) < C.shape[0]:
            # If singular, add small values to diagonal to make it invertible
            epsilon = 1e-6
            while np.linalg.matrix_rank(C) < C.shape[0]:
                C = C + np.eye(C.shape[0]) * epsilon
                epsilon *= 10
                if epsilon > 1e-3:  # Set a maximum epsilon to prevent infinite loop
                    raise ValueError("Matrix C is too close to singular and cannot be inverted")

        S = np.linalg.inv(C)
        
        # Calculate Young's moduli
        Ex = 1 / S[0, 0]
        Ey = 1 / S[1, 1]
        Ez = 1 / S[2, 2]

        # Calculate Shear moduli
        Gyz = C[3, 3]  # G23
        Gzx = C[4, 4]  # G31
        Gxy = C[5, 5]  # G12

        # Calculate Poisson's ratios
        nuxy = -S[1, 0] * Ex  # νxy = -S12/S11
        nuyx = -S[0, 1] * Ey  # νyx = -S21/S22
        nuyz = -S[2, 1] * Ey  # νyz = -S23/S22
        nuzy = -S[1, 2] * Ez  # νzy = -S32/S33
        nuzx = -S[0, 2] * Ez  # νzx = -S31/S33
        nuxz = -S[2, 0] * Ex  # νxz = -S13/S11
        
        # # Bulk modulus (K)
        # K_v = (C[0, 0] + C[1, 1] + C[2, 2] + 2*(C[0, 1] + C[1, 2] + C[2, 0])) / 9  # Voigt average
        # K_r = 1 / (S[0, 0] + S[1, 1] + S[2, 2] + 2*(S[0, 1] + S[1, 2] + S[2, 0]))  # Reuss average
        # K = (K_v + K_r) / 2  # Hill average
        
        return {
            "young's modulus": {'Ex': Ex, 'Ey': Ey, 'Ez': Ez},
            "shear modulus": {'Gyz': Gyz, 'Gzx': Gzx, 'Gxy': Gxy},
            "poisson's ratio": {'nuxy': nuxy, 'nuyx': nuyx, 'nuyz': nuyz, 'nuzy': nuzy, 'nuzx': nuzx, 'nuxz': nuxz}
        }




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
