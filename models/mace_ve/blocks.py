########################################################################
# Blocks for building GNN models
# Authors: Ivan Grega, Ilyes Batatia
# This program is distributed under the MIT License

# Some functions (e.g. TensorProductInteractions, EquivariantProductBlock) 
# were adapted from MACE (Authors: Ilyes Batatia, Gregor Simm)
########################################################################

from typing import Tuple, Optional, Union, Dict, Callable, List
from argparse import Namespace

import numpy as np
import torch
from torch_scatter import scatter
from e3nn.nn import FullyConnectedNet, Gate
from e3nn import o3
from e3nn.util.jit import compile_mode
from torch_geometric.utils import degree

from .mace import (
    BesselBasis, 
    PolynomialCutoff, 
    SymmetricContraction,
    tp_out_irreps_with_instructions,
    reshape_irreps
)
from .pna import SCALERS, AGGREGATORS

Tensor = torch.Tensor

###################
# Embedding blocks
###################
class CompleteGraph:
    """Based on L45 Practical 2
    This transform adds all pairwise edges into the edge index per data sample, 
    i.e. it builds a fully connected or complete graph
    """
    def __call__(self, edge_index, edge_feats, edge_attr, num_nodes):
        device = edge_index.device

        row = torch.arange(num_nodes, dtype=torch.long, device=device)
        col = torch.arange(num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, num_nodes).view(-1)
        col = col.repeat(num_nodes)
        _edge_index = torch.stack([row, col], dim=0)
        
        idx = edge_index[0] * num_nodes + edge_index[1]

        size = list(edge_attr.size())
        size[0] = num_nodes * num_nodes
        _edge_attr = edge_attr.new_zeros(size)
        _edge_attr[idx] = edge_attr

        size = list(edge_feats.size())
        size[0] = num_nodes * num_nodes
        _edge_feats = edge_feats.new_zeros(size)
        _edge_feats[idx] = edge_feats

        return _edge_index, _edge_feats, _edge_attr

class NodeConnectivityEmbedding(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        sender, receiver = edge_index
        uq_nodes, counts = torch.unique(receiver, return_counts=True)
        return counts.view(-1,1).float()

class RepeatNodeEmbedding(torch.nn.Module):
    def __init__(self, num_repeats: int) -> None:
        super().__init__()
        self.num_repeats = num_repeats

    def forward(
        self, 
        x: torch.Tensor # [n_nodes, @irreps]
    ) -> torch.Tensor:
        return x.tile((1, self.num_repeats)) # [n_nodes, @irreps*num_repeats]


class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(self, r_max: float, num_bessel: int, num_polynomial_cutoff: int):
        super().__init__()
        self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel, trainable=False)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
    ) -> torch.Tensor:
        bessel = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        return bessel * cutoff  # [n_edges, n_basis]


class FourierBasisEmbeddingBlock(torch.nn.Module):
    def __init__(self, n_max: int) -> None:
        super().__init__()
        
        fourier_weights = (
            np.pi
            * torch.linspace(
                start=0.0,
                end=n_max,
                steps=n_max+1,
                dtype=torch.get_default_dtype(),
            )
        )
        self.register_buffer("fourier_weights", fourier_weights)
        self.out_dim = 2*(n_max+1)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        cos_comp = torch.cos(self.fourier_weights * x)  # [n_edges, n_max+1]
        sin_comp = torch.sin(self.fourier_weights * x)  # [n_edges, n_max+1]
        return torch.cat((cos_comp, sin_comp), dim=1)   # [n_edges,2*(n_max+1)]


class PolynomialBasisEmbeddingBlock(torch.nn.Module):
    def __init__(self, max_exp: int) -> None:
        super().__init__()

        powers = torch.linspace(
            start=-max_exp,
            end=max_exp,
            steps=2*max_exp+1,
            dtype=torch.get_default_dtype()
        )
        factors = torch.tensor(3).pow(powers+1)
        self.register_buffer('powers', powers)
        self.register_buffer('factors', factors)
        self.out_dim = 2*max_exp+1

    def forward(
        self, 
        x: torch.Tensor # [n_edges, 1]
    ) -> torch.Tensor:
        return self.factors*x.pow(self.powers) # [n_edges, 2*max_exp+1]

class WaveletEmbeddingBlock(torch.nn.Module):
    def __init__(self, num_freq: int) -> None:
        super().__init__()

        fourier_weights = (
            np.pi
            * torch.linspace(
                start=1.0,
                end=num_freq,
                steps=num_freq,
                dtype=torch.get_default_dtype(),
            )
        )

        shifts = torch.zeros(num_freq, dtype=torch.get_default_dtype())

        self.register_buffer('fourier_weights', fourier_weights)
        self.shifts = torch.nn.Parameter(shifts)

        self.out_dim = 2*num_freq + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero_component = torch.exp(-0.5*x.pow(2))
        cos_components = (
            torch.cos(self.fourier_weights*(x - self.shifts))
            * torch.exp(-0.5*(x - self.shifts).pow(2))
        )
        sin_components = (
            torch.sin(self.fourier_weights*(x - self.shifts))
            * torch.exp(-0.5*(x - self.shifts).pow(2))
        )
        return torch.cat((zero_component, cos_components, sin_components), dim=1)

#################
# Readout blocks
#################

class PositiveLayer(torch.nn.Module):
    @staticmethod
    def matrix_power_2(C: torch.Tensor) -> torch.Tensor:
        return torch.linalg.matrix_power(C, 2)

    @staticmethod
    def matrix_power_4(C: torch.Tensor) -> torch.Tensor:
        return torch.linalg.matrix_power(C, 4)
    
    @staticmethod
    def matrix_exp(C: torch.Tensor) -> torch.Tensor:
        return torch.linalg.matrix_exp(C)
    
    @staticmethod
    def matrix_trunc_exp_2(C: torch.Tensor) -> torch.Tensor:
        return torch.linalg.matrix_power(torch.eye(6, device=C.device)+C/2, 2)
    
    @staticmethod
    def matrix_trunc_exp_4(C: torch.Tensor) -> torch.Tensor:
        return torch.linalg.matrix_power(torch.eye(6, device=C.device)+C/4, 4)
    
    @staticmethod 
    def identity(C: torch.Tensor) -> torch.Tensor:
        return C

    def __init__(self, params: Namespace) -> "PositiveLayer":
        super().__init__()
        
        if params.positive_function == 'matrix_power_2':
            self.func = self.matrix_power_2
        elif params.positive_function == 'matrix_power_4':
            self.func = self.matrix_power_4
        elif params.positive_function == 'matrix_exp':
            self.func = self.matrix_exp
        elif params.positive_function == 'matrix_trunc_exp_2':
            self.func = self.matrix_trunc_exp_2
        elif params.positive_function == 'matrix_trunc_exp_4':
            self.func = self.matrix_trunc_exp_4
        elif params.positive_function == 'none':
            self.func = self.identity
        else:
            raise ValueError(f'Unknown positive function: {params.positive_function}')
        
    def forward(self, C: torch.Tensor) -> torch.Tensor:
        return self.func(C)

class GeneralLinearReadoutBlock(torch.nn.Module):
    def __init__(
        self, 
        irreps_in: o3.Irreps, 
        hidden_irreps: o3.Irreps,
        irreps_out: o3.Irreps
    ):
        super().__init__()
        self.linear1 = o3.Linear(irreps_in=irreps_in, irreps_out=hidden_irreps)
        self.linear2 = o3.Linear(irreps_in=hidden_irreps, irreps_out=irreps_out)

    def forward(
        self, 
        x: torch.Tensor # [n_nodes, @irreps_in]
    ) -> torch.Tensor:  
        x = self.linear1(x)
        return self.linear2(x)  # [n_nodes, @irreps_out]


class GeneralNonLinearReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        hidden_irreps: o3.Irreps,
        irreps_out: o3.Irreps,
        gate: Callable
    ):
        super().__init__()
        self.hidden_irreps = hidden_irreps
        self.irreps_out = irreps_out
        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in hidden_irreps if ir.l == 0 and ir in self.irreps_out]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in hidden_irreps if ir.l > 0 and ir in self.irreps_out]
        )
        irreps_gates = o3.Irreps([mul, "0e"] for mul, _ in irreps_gated)
        self.equivariant_nonlin = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[gate for _, ir in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.irreps_nonlin)
        self.linear_2 = o3.Linear(
            irreps_in=self.equivariant_nonlin.irreps_out, irreps_out=self.irreps_out
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.equivariant_nonlin(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]


@compile_mode('script')
class half_irreps(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps) -> None:
        super().__init__()
        self.irreps_in = irreps_in
        out_irreps = []
        columns0 = []
        columns1 = []
        ix = 0
        for mul, ir in irreps_in:
            assert mul%2==0
            n_half = int(mul/2)
            out_irreps.append(
                (n_half, (ir.l, ir.p))
            )
            columns0.extend([i+ix for i in range(n_half*ir.dim)])
            columns1.extend([i+ix+n_half*ir.dim for i in range(n_half*ir.dim)])
            ix += mul*ir.dim

        self.irreps_out = o3.Irreps(out_irreps)
        self.register_buffer('columns_0', torch.tensor(columns0, dtype=torch.long))
        self.register_buffer('columns_1', torch.tensor(columns1, dtype=torch.long))

    def forward(
        self, 
        x: torch.Tensor # [n_nodes, @irreps]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = (x[:, self.columns_0], x[:, self.columns_1])
        return out # [n_nodes, irreps/2], # [n_nodes, @irreps/2]

class OneTPReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        
        self.split_irreps = half_irreps(irreps_in)

        irrep_half = self.split_irreps.irreps_out

        self.tp3 = o3.FullyConnectedTensorProduct(
            irreps_in1=irrep_half,
            irreps_in2=irrep_half,
            irreps_out=irreps_out,
            # path_normalization='element',
            # irrep_normalization='norm',
            internal_weights=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [num_graphs, @irreps_in]
        x0, x1 = self.split_irreps(x)
        x = self.tp3(x0,x1)
        return x # [num_graphs, @irreps_out]

# module that selects vectors with largest norm
class VectorNormSelection(torch.nn.Module):
    def __init__(self, num_vecs_in: int, num_vecs_out: int):
        super().__init__()
        self.num_vecs_out = num_vecs_out
        self.reshape = reshape_irreps(o3.Irreps(f'{num_vecs_in}x1o'))
        self.norm_fn = o3.Norm(f'{num_vecs_in}x1o', squared=True)
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        norms = self.norm_fn(x)
        indices = torch.argsort(norms, dim=1, descending=True)[:,:self.num_vecs_out].view(-1, self.num_vecs_out, 1)
        xrs = self.reshape(x)
        out = torch.gather(xrs, dim=1, index=indices.expand(-1,-1,3))
        return out

""" Indices a,b,c,d below generated as follows:
C = [['' for _ in range(6)] for _ in range(6)]
for i in range(6):
    if i<3:
        a = b = i
    elif i==3:
        a = 1; b = 2
    elif i==4:
        a = 0; b = 2
    else:
        a = 0; b = 1
    for j in range(i,6):
        if j<3:
            c = d = j
        elif j==3:
            c = 1; d = 2
        elif j==4:
            c = 0; d = 2
        else:
            c = 0; d = 1
        
        val = [a,b,c,d]
        # print(val)
        C[i][j] = val
        C[j][i] = val

rows, cols = torch.triu_indices(6,6)
a = [C[row][col][0] for row, col in zip(rows, cols)]
b = [C[row][col][1] for row, col in zip(rows, cols)]
c = [C[row][col][2] for row, col in zip(rows, cols)]
d = [C[row][col][3] for row, col in zip(rows, cols)]
print('a =', a)
print('b =', b)
print('c =', c)
print('d =', d)
"""

class Cart_4_to_Mandel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0]
        self.b = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
        self.c = [0, 1, 2, 1, 0, 0, 1, 2, 1, 0, 0, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0]
        self.d = [0, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1]

        s2 = np.sqrt(2)
        self.register_buffer(
            'mask', torch.tensor(
            [[1,1,1,s2,s2,s2],
            [1,1,1,s2,s2,s2],
            [1,1,1,s2,s2,s2],
            [s2,s2,s2,2,2,2],
            [s2,s2,s2,2,2,2],
            [s2,s2,s2,2,2,2]],
            dtype=torch.get_default_dtype(),)
        )
        rows, cols = torch.triu_indices(6,6)
        self.register_buffer(
            'rows', rows
        )
        self.register_buffer(
            'cols', cols
        )
        del rows, cols

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        C2 = C.new_zeros((C.shape[0], 6, 6))
        C2[:, self.rows, self.cols] = C[:, self.a, self.b, self.c, self.d]
        C2[:, self.cols, self.rows] = C[:, self.a, self.b, self.c, self.d]
        C2 = C2 * self.mask.view(1,6,6)
        return C2
    
class Spherical_to_Cartesian(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        formula = 'ijkl=jikl=ijlk=klij'
        indices = 'ijkl'
        rtp = o3.ReducedTensorProducts(formula, **{i: "1o" for i in indices})
        Q = rtp.change_of_basis
        Q_flat = Q.flatten(-len(indices))

        self.register_buffer('Q_flat', Q_flat)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cartesian_tensor = x @ self.Q_flat
        shape = list(x.shape[:-1]) + [3,3,3,3]
        cartesian_tensor = cartesian_tensor.view(shape)
        return cartesian_tensor

#################
# Product blocks
#################
class EquivariantProductBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: Union[int, Dict[str, int]],
        use_sc: bool = True,
    ) -> None:
        super().__init__()

        self.reshape = reshape_irreps(node_feats_irreps)

        self.use_sc = use_sc
        mul = node_feats_irreps.count(o3.Irrep(0,1))
        symmetric_contraction_out = o3.Irreps([(mul, ir) for _, ir in target_irreps])

        self.symmetric_contractions = SymmetricContraction(
            irreps_in=node_feats_irreps,
            irreps_out=symmetric_contraction_out,
            correlation=correlation,
            element_dependent=False,
            num_elements=None,
        )
        # Update linear
        self.linear = o3.Linear(
            symmetric_contraction_out,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
        self, 
        node_feats: torch.Tensor, 
        sc: torch.Tensor, 
    ) -> torch.Tensor:
        
        node_feats = self.reshape(node_feats)
        
        node_feats = self.symmetric_contractions(x=node_feats, y=None)
        if self.use_sc:
            return self.linear(node_feats) + sc

        return self.linear(node_feats)

#####################
# Interaction blocks
#####################
class TensorProductInteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        irreps_out: o3.Irreps,
        agg_norm_const: Union[float, Dict[str, float]],
        reduce: str = 'sum',
        bias: bool = False,
        MLP_dim: int = 64,
        MLP_layers: int = 3,
    ) -> None:
        super().__init__()
        self._node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self._irreps_out = irreps_out
        self.agg_norm_const = agg_norm_const
        self.reduce = reduce.lower()

        self.linear_up = o3.Linear(
            irreps_in=self._node_feats_irreps,
            irreps_out=self._node_feats_irreps,
            internal_weights=True,
            shared_weights=True
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self._node_feats_irreps,
            self.edge_attrs_irreps,
            self._irreps_out,
        )
        self.conv_tp = o3.TensorProduct(
            self._node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        # Try this initialization - includes bias term 
        layer = torch.nn.Linear(MLP_dim, self.conv_tp.weight_numel, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=10)
        self.conv_tp_weights = torch.nn.Sequential(
            torch.nn.Linear(input_dim, MLP_dim),
            torch.nn.SiLU()
        )
        for _ in range(MLP_layers-2):
            self.conv_tp_weights.append(torch.nn.Linear(MLP_dim, MLP_dim))
            self.conv_tp_weights.append(torch.nn.SiLU())
        self.conv_tp_weights.append(layer)

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.linear = o3.Linear(
            irreps_in=irreps_mid, 
            irreps_out=self._irreps_out, 
            internal_weights=True, 
            shared_weights=True,
            biases=bias # try adding bias
        )

        if self.reduce=='pna':
            self.pna = PNASimple(irreps_mid, avg_deg=agg_norm_const)

        # CHANGED
        # self.skip_tp = o3.FullyConnectedTensorProduct(
        #     self._irreps_out, '1x0e', self._irreps_out
        # )


    @property
    def irreps_in(self):
        return self._node_feats_irreps
    
    @property
    def irreps_out(self):
        return self._irreps_out

    def forward(
        self,
        node_feats: torch.Tensor, # [num_nodes, @node_feats_irreps]
        edge_attrs: torch.Tensor, # [num_edges, @edge_attrs_irreps]
        edge_feats: torch.Tensor, # [num_edges, @edge_feats_irreps]
        edge_index: torch.Tensor, # [2, num_edges]
        node_attrs: Optional[torch.Tensor] = None
    ) -> torch.Tensor: 
        sender, receiver = edge_index
        num_nodes = node_feats.shape[0]

        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(node_feats[sender], edge_attrs, tp_weights)  
        if self.reduce=='pna':
            message = self.pna(x=mji, index=receiver, dim_size=num_nodes)
        else:
            message = scatter(
                src=mji, index=receiver, dim=0, dim_size=num_nodes, reduce=self.reduce
            ) / self.agg_norm_const # [n_nodes, irreps]
        # TODO: try batchnorm
        message = self.linear(message)
        # message = self.skip_tp(message, node_attrs) # CHANGED
        return (
            message,
            None, # no skip connection
        )  # [n_nodes, channels, (lmax + 1)**2]

class EdgeUpdateBlock(torch.nn.Module):
    def __init__(
            self,
            node_ft_irreps: o3.Irreps,
            edge_sh_irreps: o3.Irreps,
            edge_ft_irreps: o3.Irreps,
    ) -> None:
        super().__init__()

        self.node_ft_irreps = node_ft_irreps
        self.edge_sh_irreps = edge_sh_irreps
        self.edge_ft_irreps = edge_ft_irreps

        self.tp_block = o3.FullyConnectedTensorProduct(
            self.node_ft_irreps,
            self.node_ft_irreps,
            self.edge_ft_irreps+self.edge_sh_irreps,
        )
        self.register_parameter('eps_ft', torch.nn.Parameter(torch.tensor(0.1)))
        self.register_parameter('eps_sh', torch.nn.Parameter(torch.tensor(0.1)))

    def forward(
            self, 
            node_ft: torch.Tensor, 
            edge_index: torch.Tensor, 
            edge_ft: torch.Tensor, 
            edge_sh: torch.Tensor
    ) -> torch.Tensor:
        sender, receiver = edge_index
        out = self.tp_block(node_ft[sender], node_ft[receiver])
        scalars = out[..., :self.edge_ft_irreps.dim]
        sh = out[..., self.edge_ft_irreps.dim:]
        edge_ft = edge_ft + self.eps_ft * scalars
        edge_sh = edge_sh + self.eps_sh * sh
        return edge_ft, edge_sh
    
#################
# Pooling blocks
#################

class GlobalSumHistoryPooling(torch.nn.Module):
    def __init__(self, reduce='sum') -> None:
        super().__init__()
        self.reduce = reduce

    def forward(
        self,
        node_ft_history: torch.Tensor, # [num_nodes, irreps, num_message_passes]
        batch_index: torch.Tensor, # [num_nodes]
        num_graphs: int,
    ) -> torch.Tensor:
        x = torch.sum(node_ft_history, dim=-1) # [num_nodes, irreps]
        graph_ft = scatter(
            src=x,
            index=batch_index.view(-1,1),
            dim=0,
            dim_size=num_graphs,
            reduce=self.reduce
        )
        return graph_ft # [irreps,]
    
class GlobalAttentionPooling(torch.nn.Module):
    # this takes calculates a magnitude per node. Does not look at separate irreps separately
    def __init__(self, irreps_in: o3.Irreps) -> None:
        super().__init__()
        self.tens_prod = o3.TensorSquare(
            irreps_in=irreps_in,
            filter_ir_out=[o3.Irrep(0,1)],
        )
        self.linear = o3.Linear(
            irreps_in=self.tens_prod.irreps_out.simplify(),
            irreps_out=o3.Irreps('1x0e')
        )

    def forward(self, 
        node_ft: Tensor, 
        batch_index: Tensor, 
        num_graphs: int
    ) -> Tensor:
        x = self.tens_prod(node_ft)
        x = torch.nn.functional.selu(x)
        x = self.linear(x) # [num_nodes, 1]
        # softmax
        exp = torch.exp(x) # [num_nodes, 1]
        z = scatter(
            src=exp,
            index=batch_index.view(-1,1),
            dim=0,
            dim_size=num_graphs,
            reduce='sum'
        ) # [num_graphs, 1]
        x = exp / z[batch_index] # [num_nodes, 1]
        x = scatter(
            src=node_ft*x.view(-1,1),
            index=batch_index.view(-1,1),
            dim=0,
            dim_size=num_graphs,
            reduce='sum'
        )
        return x
    

class IrrepBasedPooling(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps, aggr: str = 'norm_softmax') -> None:
        super().__init__()
        self.irreps = irreps
        self.expanded_irreps = o3.Irreps([(1,(ir.l,ir.p)) for mul, ir in irreps for _ in range(mul)])
        self.aggr = aggr
        if 'norm' in self.aggr:
            self.norm = o3.Norm(self.irreps, squared=False)

    def forward(self, node_ft: Tensor, batch: Tensor, num_graphs: int) -> Tensor:
        if 'norm' in self.aggr:
            norms = self.norm(node_ft)
            if 'softmax' in self.aggr:
                y = torch.exp(norms)
            elif 'softmin' in self.aggr:
                y = torch.exp(-norms)
        if 'norm' in self.aggr and 'soft' in self.aggr:
            z = scatter(y, batch, dim=0, dim_size=num_graphs, reduce='sum') # [num_graphs, num_irreps]
            y = y / z[batch] # [num_nodes, num_irreps]
            i = 0
            out = node_ft.new_zeros(node_ft.shape)
            for k, ir in self.expanded_irreps:
                out[:, i:i+ir.dim] = node_ft[:, i:i+ir.dim] * y[:,k].view(-1,1)
                i += ir.dim
            out = scatter(out, batch, dim=0, dim_size=num_graphs, reduce='sum') # [num_graphs, irreps]
        return out


# class GlobalLinearHistoryPooling(torch.nn.Module):
#     def __init__(self, num_components: int) -> None:
#         super().__init__()
#         self.num_components = num_components
#         self.register_parameter('weight', torch.nn.Parameter(torch.ones(num_components)))
    
#     def 


class GlobalElementwisePooling(torch.nn.Module):
    def __init__(self, reduce='sum') -> None:
        super().__init__()
        self.reduce = reduce

    def forward(self, node_ft: Tensor, batch: Tensor, num_graphs: int):
        return scatter(node_ft, batch, dim=-2, dim_size=num_graphs, reduce=self.reduce)


#####################
# Aggregation blocks
#####################

class PNA(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, avg_deg: Dict[str, float]) -> None:
        super().__init__()
        self.aggs = [AGGREGATORS[agg] for agg in ['mean','min','max','std']]
        self.scalers = [SCALERS[scale] for scale in ['identity','amplification','attenuation']]
        self.avg_deg = avg_deg

        irreps_lin_in = o3.Irreps([(12*mul,(ir.l,ir.p)) for mul,ir in irreps_in])
        self.post_pn = o3.Linear(irreps_in=irreps_lin_in, irreps_out=irreps_in)

    def forward(self, x: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
        outs = [aggr(x, index, dim_size) for aggr in self.aggs]
        out = torch.stack(outs, dim=-1)

        deg = degree(index, dim_size, dtype=x.dtype).view(-1, 1)
        outs = []
        for k in range(out.shape[-1]):
            for scaler in self.scalers:
                x = out[:,:,k]
                outs.append(scaler(x, deg, self.avg_deg))
        # outs = [scaler(out, deg, self.avg_deg) for scaler in self.scalers]
        out = torch.cat(outs, dim=-1)

        out = torch.flatten(out, start_dim=1)
        out = self.post_pn(out)

        return out
    
class PNA_Irreps(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()

        self.mean_agg = AGGREGATORS['mean']
        self.max_agg = IrrepBasedPooling(irreps, 'norm_softmax')
        self.min_agg = IrrepBasedPooling(irreps, 'norm_softmin')

        temp_irreps = irreps*3
        slices = temp_irreps.slices()
        temp_irreps, _, inv = temp_irreps.sort()
        temp_irreps = temp_irreps.simplify()
        self.sorting_slices = [slices[i] for i in inv]

        self.linear = o3.Linear(temp_irreps, irreps, 
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

    def forward(self, x: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
        mean = self.mean_agg(x, index, dim_size)
        max = self.max_agg(x, index, dim_size)
        min = self.min_agg(x, index, dim_size)

        out = torch.cat([mean, max, min], dim=1)
        out = torch.cat([out[:, s] for s in self.sorting_slices], dim=1)
        out = self.linear(out)
        return out


class PNASimple(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, avg_deg: Dict[str, float]) -> None:
        super().__init__()
        # aggregs = ['sum','mean']
        aggregs = ['mean','min','max','std']
        scalers = ['identity','amplification','attenuation']

        # self.norm_fn = o3.Norm(irreps_in, squared=True)
        self.irreps_in = irreps_in
        self.aggs = [AGGREGATORS[agg] for agg in aggregs]
        self.scalers = [SCALERS[scale] for scale in scalers]
        self.avg_deg = avg_deg

        self.post_pn = torch.nn.Linear(len(aggregs)*len(scalers),1, bias=False)

    def forward(self, x: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
        # norms = self.norm_fn(x)
        # outs = [aggr(x, index, dim_size, norms=norms, irreps=self.irreps_in) for aggr in self.aggs]
        outs = [aggr(x, index, dim_size) for aggr in self.aggs]
        out = torch.stack(outs, dim=-1)

        deg = degree(index, dim_size, dtype=x.dtype).view(-1, 1)
        outs = []
        for k in range(out.shape[-1]):
            for scaler in self.scalers:
                x = out[:,:,k]
                outs.append(scaler(x, deg, self.avg_deg))
        out = torch.stack(outs, dim=2)

        out = self.post_pn(out)
        out = out.squeeze(2)
        return out

#############
# GNN Layers
#############

class GraphAttention(torch.nn.Module):
    def __init__(
        self, 
        input_irreps: o3.Irreps,
        query_irreps: o3.Irreps,
        key_irreps: o3.Irreps,
        output_irreps: o3.Irreps,
        edge_sh_irreps: o3.Irreps,
        edge_scalar_basis: int
    ) -> None:
        super().__init__()

        self.h_q = o3.Linear(input_irreps, query_irreps)
        self.tp_k = o3.FullyConnectedTensorProduct(
            input_irreps, edge_sh_irreps, key_irreps, shared_weights=False
        )
        self.fc_k = FullyConnectedNet(
            [edge_scalar_basis, 32, self.tp_k.weight_numel], 
            act=torch.nn.functional.silu
        )
        self.tp_v = o3.FullyConnectedTensorProduct(
            input_irreps, edge_sh_irreps, output_irreps, shared_weights=False
        )
        self.fc_v = FullyConnectedNet(
            [edge_scalar_basis, 32, self.tp_v.weight_numel], 
            act=torch.nn.functional.silu
        )
        self.dot = o3.FullyConnectedTensorProduct(
            query_irreps, key_irreps, "0e"
        )

    def forward(
        self,
        node_ft: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_scalars: torch.Tensor,
    ) -> torch.Tensor:
        sender, receiver = edge_index
        q = self.h_q(node_ft)
        k = self.tp_k(node_ft[sender], edge_sh, self.fc_k(edge_scalars))
        v = self.tp_v(node_ft[sender], edge_sh, self.fc_v(edge_scalars))

        exp = self.dot(q[receiver], k).exp()
        z = scatter(exp, receiver, dim=0, dim_size=node_ft.shape[0])
        alpha = exp / z[receiver]
        return scatter(alpha.relu().sqrt() * v, receiver, dim=0, dim_size=node_ft.shape[0])

class MACELayer(torch.nn.Module):
    def __init__(
        self,
        input_irreps: o3.Irreps,
        edge_sh_irreps: o3.Irreps,
        edge_scalars_irreps: o3.Irreps,
        interaction_irreps: o3.Irreps,
        output_irreps: o3.Irreps,
        interaction_agg_norm_const: Union[float, Dict],
        interaction_reduction: str,
        interaction_bias: bool,
        product_correlation: int,
        MLP_dim: int = 64,
        MLP_layers: int = 3,
    ) -> None:
        super().__init__()
        
        
        self.interaction = TensorProductInteractionBlock(
            node_feats_irreps=input_irreps,
            edge_attrs_irreps=edge_sh_irreps,
            edge_feats_irreps=edge_scalars_irreps,
            irreps_out=interaction_irreps,
            agg_norm_const=interaction_agg_norm_const,
            reduce=interaction_reduction,
            bias=interaction_bias,
            MLP_dim=MLP_dim,
            MLP_layers=MLP_layers
        )

        self.product = EquivariantProductBlock(
            node_feats_irreps=self.interaction.irreps_out,
            target_irreps=output_irreps,
            correlation=product_correlation,
            use_sc=False
        )

    def forward(
        self,
        node_ft: Tensor, 
        edge_index: Tensor, 
        edge_sh: Tensor, 
        edge_scalars: Tensor
    ) -> Tensor:
        node_ft, sc = self.interaction(node_ft, edge_sh, edge_scalars, edge_index)
        return self.product(node_ft, sc)

class CGCLayer(torch.nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, reduction: str = 'sum') -> None:
        super().__init__()
        
        self.num_hid_dim = 2*node_dim+edge_dim
        self.fc_values = torch.nn.Linear(self.num_hid_dim, node_dim)
        self.fc_multip = torch.nn.Linear(self.num_hid_dim, node_dim)

        self.reduction = reduction

    def forward(self, x: Tensor, edge_index: Tensor, edge_ft: Tensor):
        sender, receiver = edge_index

        c = torch.cat([x[sender], x[receiver], edge_ft], dim=1) # [num_edges, num_hid_dim]

        msg = torch.nn.functional.softplus(self.fc_values(c))*torch.sigmoid(self.fc_multip(c))

        return scatter(msg, receiver, dim=0, dim_size=x.shape[0], reduce=self.reduction)
