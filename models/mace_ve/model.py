from typing import Any, Optional, Tuple, Dict, Union
from argparse import Namespace

import torch
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.data import Batch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.io import CartesianTensor
from torch import nn

from .blocks import (
    MACELayer,
    Cart_4_to_Mandel,
    GeneralNonLinearReadoutBlock,
    GeneralLinearReadoutBlock,
    OneTPReadoutBlock,
    GlobalSumHistoryPooling,
    TensorProductInteractionBlock,
    EquivariantProductBlock,
    Spherical_to_Cartesian,
    PositiveLayer
)
from .mace import get_edge_vectors_and_lengths, reshape_irreps

class GNN_Head(torch.nn.Module):
    def __init__(self, params: Namespace) -> "GNN_Head":
        super().__init__()
        self.params = params

        hidden_irreps = o3.Irreps(params.hidden_irreps)

        self.number_of_edge_basis = params.num_edge_bases
        node_ft_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        edge_feats_irreps = o3.Irreps(f"{self.number_of_edge_basis*2}x0e")
        edge_attr_irreps = o3.Irreps.spherical_harmonics(params.lmax)
     
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (edge_attr_irreps * num_features).sort()[0].simplify()
        readout_irreps = o3.Irreps(params.readout_irreps)

        self.num_interactions = params.message_passes
        self.layers = torch.nn.ModuleList([MACELayer(
            node_ft_irreps,
            edge_attr_irreps,
            edge_feats_irreps,
            interaction_irreps,
            hidden_irreps,
            params.agg_norm_const,
            params.interaction_reduction,
            True, 
            params.correlation,
            MLP_dim=params.inter_MLP_dim,
            MLP_layers=params.inter_MLP_layers,
        )])
        for _ in range(self.num_interactions-1):
            self.layers.append(
                MACELayer(
                    hidden_irreps,
                    edge_attr_irreps,
                    edge_feats_irreps,
                    interaction_irreps,
                    hidden_irreps,
                    params.agg_norm_const,
                    params.interaction_reduction,
                    True, 
                    params.correlation,
                    MLP_dim=params.inter_MLP_dim,
                    MLP_layers=params.inter_MLP_layers,
                )
            )
       
        self.nonlin_readout = GeneralNonLinearReadoutBlock(
                irreps_in=hidden_irreps,
                irreps_out=readout_irreps,
                hidden_irreps=hidden_irreps,
                gate=torch.nn.functional.silu,
            )  
       
        self.global_reduction = params.global_reduction
        irreps_post_linear = o3.Irreps('2x0e+2x2e+1x4e')
        self.linear = o3.Linear(readout_irreps, irreps_post_linear, 
            internal_weights=True, 
            shared_weights=True,
            biases=True
        )


        self.sph_to_cart = Spherical_to_Cartesian()
        self.cart_to_Mandel = Cart_4_to_Mandel()
        self.positive_layer = PositiveLayer(params)

    def forward(self, edge_index: Tensor, node_ft: Tensor, edge_sh: Tensor, edge_feats: Tensor, batch_idx: Tensor, num_graphs: int) -> Tensor:

        node_ft = self.layers[0](node_ft, edge_index, edge_sh, edge_feats)
        for i in range(1, self.num_interactions):
            node_ft = node_ft + self.layers[i](node_ft, edge_index, edge_sh, edge_feats)
       
        output = self.nonlin_readout(node_ft)

        graph_ft = scatter(
            src=output,
            index=batch_idx,
            dim=0,
            dim_size=num_graphs,
            reduce=self.global_reduction
        )

        stiff_ft = self.linear(graph_ft)
        stiff = self.sph_to_cart(stiff_ft)
        C = self.cart_to_Mandel(stiff)
        C = self.positive_layer(C)
        return C


class GNN_Reg_Head(torch.nn.Module):
    def __init__(self, params: Namespace) -> "GNN_Head":
        super().__init__()
        self.params = params

        hidden_irreps = o3.Irreps(params.hidden_irreps)

        self.number_of_edge_basis = params.num_edge_bases
        node_ft_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        edge_feats_irreps = o3.Irreps(f"{self.number_of_edge_basis * 2}x0e")
        edge_attr_irreps = o3.Irreps.spherical_harmonics(params.lmax)

        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (edge_attr_irreps * num_features).sort()[0].simplify()
        readout_irreps = o3.Irreps(params.readout_irreps)

        self.num_interactions = params.message_passes
        self.layers = torch.nn.ModuleList([MACELayer(
            node_ft_irreps,
            edge_attr_irreps,
            edge_feats_irreps,
            interaction_irreps,
            hidden_irreps,
            params.agg_norm_const,
            params.interaction_reduction,
            True,
            params.correlation,
            MLP_dim=params.inter_MLP_dim,
            MLP_layers=params.inter_MLP_layers,
        )])
        for _ in range(self.num_interactions - 1):
            self.layers.append(
                MACELayer(
                    hidden_irreps,
                    edge_attr_irreps,
                    edge_feats_irreps,
                    interaction_irreps,
                    hidden_irreps,
                    params.agg_norm_const,
                    params.interaction_reduction,
                    True,
                    params.correlation,
                    MLP_dim=params.inter_MLP_dim,
                    MLP_layers=params.inter_MLP_layers,
                )
            )

        self.nonlin_readout = GeneralNonLinearReadoutBlock(
            irreps_in=hidden_irreps,
            irreps_out=readout_irreps,
            hidden_irreps=hidden_irreps,
            gate=torch.nn.functional.silu,
        )

        self.global_reduction = params.global_reduction
        irreps_post_linear = o3.Irreps('2x0e+2x2e+1x4e')
        self.linear = o3.Linear(readout_irreps, irreps_post_linear,
                                internal_weights=True,
                                shared_weights=True,
                                biases=True
                                )

        self.regression_head = nn.Sequential(
            nn.Linear(irreps_post_linear.dim, 64),  # Adjust dimensions accordingly
            nn.ReLU(),
            nn.Linear(64, params.output_dim)  # Output dimension matches the number of mechanical properties you want to predict
        )


        # self.sph_to_cart = Spherical_to_Cartesian()
        # self.cart_to_Mandel = Cart_4_to_Mandel()
        # self.positive_layer = PositiveLayer(params)

    def forward(self, edge_index: Tensor, node_ft: Tensor, edge_sh: Tensor, edge_feats: Tensor, batch_idx: Tensor,
                num_graphs: int) -> Tensor:

        node_ft = self.layers[0](node_ft, edge_index, edge_sh, edge_feats)
        for i in range(1, self.num_interactions):
            node_ft = node_ft + self.layers[i](node_ft, edge_index, edge_sh, edge_feats)

        output = self.nonlin_readout(node_ft)

        graph_ft = scatter(
            src=output,
            index=batch_idx,
            dim=0,
            dim_size=num_graphs,
            reduce=self.global_reduction
        )

        stiff_ft = self.linear(graph_ft)
        C = self.regression_head(stiff_ft)
        # stiff = self.sph_to_cart(stiff_ft)
        # C = self.cart_to_Mandel(stiff)
        # C = self.positive_layer(C)
        return C



class EnergyEquivGNN(torch.nn.Module):
    def __init__(self, params: Namespace, *args: Any, **kwargs: Any) -> "EnergyEquivGNN":
        super().__init__(*args, **kwargs)

        self.params = params
        hidden_irreps = o3.Irreps(params.hidden_irreps)

        self.node_ft_embedding = torch.nn.Linear(in_features=params.node_ft_dim, out_features=hidden_irreps.count(o3.Irrep(0,1)))
        self.number_of_edge_basis = params.num_edge_bases
        self.max_edge_radius = params.max_edge_radius
        edge_attr_irreps = o3.Irreps.spherical_harmonics(params.lmax)
        self.spherical_harmonics = o3.SphericalHarmonics(
            edge_attr_irreps,
            normalize=True, normalization='component'
        )

        # self.stiffness_head = GNN_Head(params)
        self.stiffness_head = GNN_Reg_Head(params)
        # another head can be added to do compliance prediction
        # self.compliance_head = GNN_Head(params)
     


    def forward(self, positions, edge_index, shifts, edge_attr, node_feat, num_graphs, batch) -> Dict:
        
        edge_index = edge_index
        node_ft = node_feat

        node_ft = self.node_ft_embedding(node_ft)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=positions, edge_index=edge_index, shifts=shifts
        )
        edge_length_embedding = soft_one_hot_linspace(
            lengths.squeeze(-1), start=0, end=0.6, number=self.number_of_edge_basis, basis='gaussian', cutoff=False
        )
        edge_radii = edge_attr
        edge_radius_embedding = soft_one_hot_linspace(
            edge_radii.squeeze(-1), 0, self.max_edge_radius, self.number_of_edge_basis, 'gaussian', False
        )
        edge_feats = torch.cat(
            (edge_length_embedding, edge_radius_embedding), 
            dim=1
        )
        edge_sh = self.spherical_harmonics(vectors)
        
        C_pos = self.stiffness_head(edge_index, node_ft, edge_sh, edge_feats, batch, num_graphs)

        return {'y': C_pos}