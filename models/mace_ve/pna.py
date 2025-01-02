###############################################################################
# Principal Neighbourhood Aggregation 
# Code copied from pna code https://github.com/lukecavabarrett/pna.git
# Authors: Gabriele Corso, Luca Cavalleri
# This program is distributed under the MIT License

# Edits by Ivan Grega to enable the use with e3nn and maintain equivariance
###############################################################################
from typing import Dict, Optional, Callable

import torch
from torch import Tensor
from torch_scatter import scatter, scatter_min, scatter_max

from e3nn import o3

# Implemented with the help of Matthias Fey, author of PyTorch Geometric
# For an example see https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pna.py

def scale_identity(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src


def scale_amplification(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src * (torch.log(deg + 1) / avg_deg['log'])


def scale_attenuation(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    scale = avg_deg['log'] / torch.log(deg + 1)
    scale[deg == 0] = 1
    return src * scale


def scale_linear(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src * (deg / avg_deg['lin'])


def scale_inverse_linear(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    scale = avg_deg['lin'] / deg
    scale[deg == 0] = 1
    return src * scale


SCALERS = {
    'identity': scale_identity,
    'amplification': scale_amplification,
    'attenuation': scale_attenuation,
    'linear': scale_linear,
    'inverse_linear': scale_inverse_linear
}

# Implemented with the help of Matthias Fey, author of PyTorch Geometric
# For an example see https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pna.py

def aggregate_sum(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='sum')


def aggregate_mean(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='mean')


def aggregate_min(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='min')


def aggregate_max(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='max')


def aggregate_var(src, index, dim_size):
    mean = aggregate_mean(src, index, dim_size)
    mean_squares = aggregate_mean(src * src, index, dim_size)
    return mean_squares - mean * mean


def aggregate_std(src, index, dim_size):
    return torch.sqrt(torch.relu(aggregate_var(src, index, dim_size)) + 1e-5)


def aggregate_sum_irreps(src: Tensor, index: Tensor, dim_size: int, norms: Tensor, irreps: o3.Irreps):
    return scatter(src, index, 0, None, dim_size, reduce='sum')


def aggregate_mean_irreps(src: Tensor, index: Tensor, dim_size: int, norms: Tensor, irreps: o3.Irreps):
    return scatter(src, index, 0, None, dim_size, reduce='mean')


def aggregate_min_irreps(src: Tensor, index: Tensor, dim_size: int, norms: Tensor, irreps: o3.Irreps):
    irreps_expanded = o3.Irreps([(1,(ir.l,ir.p)) for mul, ir in irreps for _ in range(mul)])
    _, inds = scatter_min(norms, index, 0, None, dim_size)
    out = torch.empty((dim_size, src.shape[1]), dtype=src.dtype, device=src.device)
    i = 0
    for norm_col, ir in enumerate(irreps_expanded):
        col = i + ir.dim
        out[:, i:col] = src[inds[:,norm_col], i:col]
        i = col
    return out

def aggregate_max_irreps(src: Tensor, index: Tensor, dim_size: int, norms: Tensor, irreps: o3.Irreps):
    irreps_expanded = o3.Irreps([(1,(ir.l,ir.p)) for mul, ir in irreps for _ in range(mul)])
    _, inds = scatter_max(norms, index, 0, None, dim_size)
    out = torch.empty((dim_size, src.shape[1]), dtype=src.dtype, device=src.device)
    i = 0
    for norm_col, ir in enumerate(irreps_expanded):
        col = i + ir.dim
        out[:, i:col] = src[inds[:,norm_col], i:col]
        i = col
    return out


def aggregate_std_irreps(src: Tensor, index: Tensor, dim_size: int, norms: Tensor, irreps: o3.Irreps):
    return torch.sqrt(torch.relu(aggregate_var(src, index, dim_size)) + 1e-5)

AGGREGATORS = {
    'sum': aggregate_sum,
    'mean': aggregate_mean,
    'min': aggregate_min,
    'max': aggregate_max,
    'std': aggregate_std,
    'min_irreps': aggregate_min_irreps,
    'max_irreps': aggregate_max_irreps,
}