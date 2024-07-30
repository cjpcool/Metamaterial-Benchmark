import numpy as np
import torch
from torch_scatter import scatter
from ase import Atoms
from ase.neighborlist import NeighborList


class Structure:
    def __init__(self, lattice, edge_index, coordinates, properties=None, properties_names=None, diameter=None,
                 is_cartesian=True):
        '''

        :param lattice:
        :param edge_index:
        :param coordinates:
        :param properties:
        :param properties_names:
        :param diameter:
        :param is_cartesian:
        :param node_types: None or torch.LongTensor[{0,1},atom_type], 1 denotes is an atom,
        '''
        if lattice.shape == (3, 3):
            self.lattice_vector = lattice
            self.lattice_params = self.lattice_vector_to_parameters(self.lattice_vector)
        if lattice.shape == (2, 3):
            self.lattice_vector = self.lattice_params_to_matrix_torch(lattice[0], lattice[1])
            self.lattice_params = lattice

        self.edge_index = edge_index.t()

        self.num_edges = self.edge_index.shape[1]

        self.num_nodes = coordinates.shape[0]
        self.diameter = diameter
        if is_cartesian:
            self.frac_coords = self.cart_to_frac_coords(coordinates, self.lattice_params[0], self.lattice_params[1],
                                                        self.num_nodes)
            self.cart_coords = coordinates
        else:
            self.cart_coords = self.frac_to_cart_coords(coordinates, self.lattice_params[0], self.lattice_params[1],
                                                        self.num_nodes)
            self.frac_coords = coordinates

        self.properties_names = properties_names
        self.properties = properties
        self.to_jimages = self.calculate_to_jimages_efficient(self.cart_coords.numpy(), self.edge_index.numpy(), self.lattice_vector.numpy())



    @staticmethod
    def calculate_to_jimages_efficient(coordinates, edge_index, lattice_vectors):
        """
        """

        def find_images_and_correct_vector(vector, lattice_inv, lattice_vectors):
            images = np.rint(np.dot(lattice_inv, vector))
            corrected_vector = vector - np.dot(images, lattice_vectors)
            return images.astype(int), corrected_vector

        lattice_inv = np.linalg.inv(lattice_vectors)

        num_edges = edge_index.shape[1]
        to_jimages = np.zeros((num_edges, 3), dtype=int)

        for idx, (i, j) in enumerate(edge_index.T):
            vector_ij = coordinates[j] - coordinates[i]
            jimages, corrected_vector = find_images_and_correct_vector(vector_ij, lattice_inv, lattice_vectors)

            to_jimages[idx] = jimages

        return torch.tensor(to_jimages)


    @staticmethod
    def calculate_to_jimages_ase(coordinates, edge_index, lattice_lengths, lattice_angles, is_cartesian=True):
        """
        使用ASE库计算to_jimages，考虑晶格角度。

        参数:
        - coordinates: 原子坐标，可以是分数坐标（fractional）或笛卡尔坐标（cartesian）。
        - edge_index: 形状为(2, num_edges)的二维数组，表示边的连接。
        - lattice_lengths: 晶格向量的长度(a, b, c)。
        - lattice_angles: 晶格向量间的夹角(alpha, beta, gamma)。

        返回:
        - to_jimages: 形状为(num_edges, 3)的二维数组，每行为对应边的to_jimages。
        """
        # 使用cellpar参数来包含晶格长度和角度
        cell_parameters = lattice_lengths.tolist() + lattice_angles.tolist()

        # 根据坐标类型创建ASE的Atoms对象
        if is_cartesian:  # is_cartesian
            atoms = Atoms(positions=coordinates, cell=cell_parameters, pbc=True)
        else:  # fractional
            atoms = Atoms(positions=coordinates, cell=cell_parameters, pbc=True, scale_positions=True)

        cutoff = max(lattice_lengths) * 2  # approximate
        nl = NeighborList(cutoffs=[cutoff / 2.] * len(atoms), self_interaction=False, bothways=True, skin=0.)
        nl.update(atoms)

        to_jimages = []
        for idx_i in range(len(atoms)):
            indices, offsets = nl.get_neighbors(idx_i)
            for idx_j, offset in zip(indices, offsets):
                if (idx_i, idx_j) in edge_index.T.tolist() or (idx_j, idx_i) in edge_index.T.tolist():
                    to_jimages.append(offset)

        to_jimages_array = np.array(to_jimages).reshape((-1, 3))

        return torch.tensor(to_jimages_array)

    @staticmethod
    def lattice_vector_to_parameters(lattice_vector):
        """
        lengths (list): 晶格向量的长度。
        angles (list): 晶格向量之间的角度（度）。
        """
        # 确保向量为numpy数组并计算长度
        a, b, c = lattice_vector[0], lattice_vector[1], lattice_vector[2]
        lengths = [torch.norm(a), torch.norm(b), torch.norm(c)]

        dot_ab = torch.dot(b, c)
        dot_ac = torch.dot(a, c)
        dot_bc = torch.dot(a, b)
        alpha = torch.acos(dot_ab / (lengths[1] * lengths[2]))
        beta = torch.acos(dot_ac / (lengths[0] * lengths[2]))
        gamma = torch.acos(dot_bc / (lengths[0] * lengths[1]))

        angles = torch.tensor([alpha * 180 / torch.pi, beta * 180 / torch.pi, gamma * 180 / torch.pi])

        return torch.stack(lengths), angles

    @staticmethod
    def lattice_params_to_matrix_torch(lengths, angles):
        """Batched torch version to compute lattice matrix from params.

        lengths: torch.Tensor of shape (N, 3), unit A
        angles: torch.Tensor of shape (N, 3), unit degree
        """
        angles = torch.unsqueeze(angles, dim=0)
        lengths = torch.unsqueeze(lengths, dim=0)

        angles_r = torch.deg2rad(angles)
        coses = torch.cos(angles_r)
        sins = torch.sin(angles_r)

        val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
        # Sometimes rounding errors result in values slightly > 1.
        val = torch.clamp(val, -1., 1.)
        gamma_star = torch.arccos(val)

        vector_a = torch.stack([
            lengths[:, 0] * sins[:, 1],
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 0] * coses[:, 1]], dim=1)
        vector_b = torch.stack([
            -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
            lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
            lengths[:, 1] * coses[:, 0]], dim=1)
        vector_c = torch.stack([
            torch.zeros(lengths.size(0), device=lengths.device),
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 2]], dim=1)
        return torch.stack([vector_a, vector_b, vector_c], dim=1)

    @staticmethod
    def frac_to_cart_coords(
            frac_coords,
            lengths,
            angles,
            num_atoms,
    ):
        lattice = Structure.lattice_params_to_matrix_torch(lengths, angles)
        lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
        cart_coords = torch.einsum('bi,bij->bj', frac_coords.float(), lattice_nodes.float())  # cart coords
        return cart_coords

    @staticmethod
    def cart_to_frac_coords(
            cart_coords,
            lengths,
            angles,
            num_atoms,
    ):
        lattice = Structure.lattice_params_to_matrix_torch(lengths, angles)
        # use pinv in case the predicted lattice is not rank 3
        inv_lattice = torch.pinverse(lattice)
        inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)
        frac_coords = torch.einsum('bi,bij->bj', cart_coords.float(), inv_lattice_nodes.float())
        return frac_coords

    @staticmethod
    def correct_cart_coords(cart_coords, lengths, angles, num_atoms, batch):
        lattice = Structure.lattice_params_to_matrix_torch(lengths, angles)
        lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)

        inv_lattice = torch.inverse(lattice)
        inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)

        frac_coords = torch.einsum('bi,bij->bj', cart_coords, inv_lattice_nodes)
        frac_coords = Structure.correct_frac_coords(frac_coords, batch)

        cart_coords = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)  # cart coords
        return cart_coords

    @staticmethod
    def correct_frac_coords(frac_coords, batch):
        new_frac_coords = (frac_coords + 0.5) % 1. - 0.5
        min_frac_coords = scatter(new_frac_coords, batch, dim=0, reduce='min')
        max_frac_coords = scatter(new_frac_coords, batch, dim=0, reduce='max')
        offset_frac_coords = (min_frac_coords + max_frac_coords) / 2.0
        new_frac_coords = new_frac_coords - offset_frac_coords[batch]

        return new_frac_coords