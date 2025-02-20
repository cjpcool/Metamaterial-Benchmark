from itertools import combinations

import torch
import numpy as np




def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)

def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])

def ndarry_to_tensor(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x

def cart_to_frac_coords(
        cart_coords,
        lattice,
        num_atoms,
):
    cart_coords = ndarry_to_tensor(cart_coords)
    lattice = ndarry_to_tensor(lattice)
    num_atoms = ndarry_to_tensor(num_atoms)

    if len(lattice.shape) == 2:
        lattice = lattice.unsqueeze(0)
    # use pinv in case the predicted lattice is not rank 3
    inv_lattice = torch.pinverse(lattice)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)
    frac_coords = torch.einsum('bi,bij->bj', cart_coords.float(), inv_lattice_nodes.float())
    return frac_coords.cpu().numpy()



def frac_to_cart_coords(
        frac_coords,
        lattice,
        num_atoms,
):
    frac_coords = ndarry_to_tensor(frac_coords)
    lattice = ndarry_to_tensor(lattice)
    num_atoms = ndarry_to_tensor(num_atoms)

    if len(lattice.shape) == 2:
        lattice = lattice.unsqueeze(0)
    lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
    cart_coords = torch.einsum('bi,bij->bj', frac_coords.float(), lattice_nodes.float())  # cart coords
    return cart_coords.cpu().numpy()







# With Bugs, DO NOT USE
def find_lattice_vectors(cart_coords):
    '''
    # With Bugs, DO NOT USE
    Args:
        cart_coords:

    Returns:

    '''
    N = cart_coords.shape[0]

    possible_lvs = set()

    for i, j in combinations(range(N), 2):
        diff = tuple(cart_coords[j] - cart_coords[i])
        possible_lvs.add(diff)

    possible_lvs = np.array(list(possible_lvs))

    valid_lvs = []

    for lv_combination in combinations(possible_lvs, 3):
        lv_matrix = np.array(lv_combination)

        # 检查这三个向量是否线性无关（行列式不为零）
        if np.linalg.det(lv_matrix) != 0:
            # 构建一个矩阵，其中每一行都是一个节点坐标与原点之间的向量
            relative_coords = cart_coords - cart_coords.min(axis=0)

            # 计算相对坐标在晶格向量基下的系数
            try:
                coeffs = np.linalg.solve(lv_matrix.T, relative_coords.T).T
            except np.linalg.LinAlgError:
                # 如果矩阵是奇异的，跳过这个组合
                continue

            if np.allclose(coeffs, coeffs.round()):
                valid_lvs.append(lv_matrix)

    # 选择第一个有效的晶格向量组合作为结果
    if valid_lvs:
        lattice_vectors = valid_lvs[0]
    else:
        raise ValueError("No valid lattice vectors found.")

    return lattice_vectors



