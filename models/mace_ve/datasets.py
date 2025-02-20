import os
import os.path as osp
import shutil
import sys
from typing import Callable, List, Optional, Union, Iterable, Tuple
from random import shuffle
import logging
from multiprocessing import Pool

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm
import pandas as pd

from torch_geometric.data import (
    Data,
    InMemoryDataset
)

from lattices.lattices import Lattice
from lattices.lattices import Catalogue
from lattices.lattices import elasticity_func

class GLAMM_Dataset(InMemoryDataset):
    r"""Lattice dataset.
    Work in progress.

    """  # noqa: E501


    def __init__(self, 
            root: str, 
            catalogue_path: str,
            dset_fname: str,
            representation: str = 'fund_inner',
            node_ft: str = 'ones',
            edge_ft: str = 'r',
            graph_ft_format: str = 'cartesian_4',
            n_reldens: int = 1,
            choose_reldens: str = 'first',
            multiprocessing: Optional[Union[bool, int]] = False,
            regex_filter: Optional[str] = None,
            #
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
        ):
        
       
        self.graph_ft_format = graph_ft_format

        self.nreldens  = n_reldens
        if choose_reldens=='first':
            self.reldens_slice = slice(None, n_reldens, 1)
        elif choose_reldens=='last':
            self.reldens_slice = slice(-n_reldens, None, 1)
        elif choose_reldens=='half':
            self.reldens_slice = slice(None, 2*n_reldens, 2)
        elif choose_reldens=='all':
            self.reldens_slice = slice(None, None, 1)
        else:
            raise ValueError(f'choose_reldens `{choose_reldens}` not recognised')

        if representation in ['fund_inner']:
            self.repr = representation
        else:
            raise ValueError(f'Representation {representation} does not exist')

        if node_ft in ['ones']:
            self.node_ft_format = node_ft
        else:
            raise ValueError(f'Node ft format `{node_ft}` not recognised')

        for key in edge_ft.split(','):
            if key not in ['L','r','e_vec','euler']:
                raise ValueError(f'Edge feature format key `{key}` not recognised')
        self.edge_ft_format = edge_ft
            
        self.multiprocessing = multiprocessing
                
        self.catalogue_path = osp.realpath(catalogue_path)
        self.catalogue_name = osp.basename(catalogue_path)
        self.processed_name = osp.basename(dset_fname)
        self.regex_filter = regex_filter

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    @property
    def raw_file_names(self) -> List[str]:
        return [self.catalogue_name]


    @property
    def processed_file_names(self) -> str:
        return self.processed_name
       

    def download(self):
        assert osp.isfile(self.catalogue_path), f'Catalogue file {self.catalogue_path} does not exist'
        shutil.copy(self.catalogue_path, self.raw_dir)
        # for url in [self.raw_url0]:
        #     file_path = download_url(url, self.raw_dir)
        #     extract_gz(osp.join(self.raw_dir, self.raw_compressed_name), self.raw_dir)

    @staticmethod
    def process_one(
            lat_data: dict,
            edge_ft_format: str = 'r',
            graph_ft_format: str = 'cartesian_4',
            reldens_slice: slice = slice(None, None, 1),
            pre_filter: Callable = None,
            pre_transform: Callable = None,
        ):
        name = lat_data['name']
        if 'nodal_positions' in lat_data:
            nodal_positions = np.atleast_2d(lat_data['nodal_positions'])
        elif 'reduced_node_coordinates' in lat_data:
            nodal_positions = np.atleast_2d(lat_data['reduced_node_coordinates'])
        else:
            raise ValueError('No nodal positions found')
        fundamental_edge_adjacency = np.atleast_2d(lat_data['fundamental_edge_adjacency'])
        fundamental_tess_vecs = np.atleast_2d(lat_data['fundamental_tesselation_vecs'])
        lattice_constants = np.array(lat_data['lattice_constants'])
        if 'compliance_tensors_M' in lat_data:
            compliance_tensors = lat_data['compliance_tensors_M']
            compliance_tensors_M = compliance_tensors
        elif 'compliance_tensors_V' in lat_data:
            compliance_tensors = lat_data['compliance_tensors_V']
            # Convert from Voigt to Mandel
            try:
                compliance_tensors_M = {k:elasticity_func.compliance_Voigt_to_Mandel(v) for k,v in compliance_tensors.items()}
            except TypeError:
                logging.warning(f'Failed to convert compliance tensors for {name}')
                compliance_tensors_M = {k:None for k,v in compliance_tensors.items()}
            compliance_tensors = compliance_tensors_M

        uq_inds = np.unique(fundamental_edge_adjacency)
        nodal_positions = nodal_positions[uq_inds]
        edge_adjacency = np.searchsorted(uq_inds, fundamental_edge_adjacency)
        if fundamental_tess_vecs.shape[1]==6:
            tessellation_vecs = fundamental_tess_vecs[:, 3:] - fundamental_tess_vecs[:, :3]
        elif fundamental_tess_vecs.shape[1]==3:
            tessellation_vecs = fundamental_tess_vecs
        else:
            raise ValueError(f'Fundamental tessellation vectors shape {fundamental_tess_vecs.shape} not recognised')
        unit_shifts = tessellation_vecs.astype(int)

        # transform coordinates
        Q = Lattice.calculate_transform_matrix(lattice_constants)
        nodal_positions = nodal_positions@(Q.T)
        tessellation_vecs = tessellation_vecs@(Q.T)

        edge_adjacency = np.row_stack(
            (edge_adjacency, edge_adjacency[:,::-1])
            ) # reverse connections
        unit_shifts = np.row_stack(
            (unit_shifts, -unit_shifts)
        )
        tessellation_vecs = np.row_stack(
            (tessellation_vecs, -tessellation_vecs)
        )

        # data for strut thickness calibration
        edge_vecs = nodal_positions[edge_adjacency[:,1]] - nodal_positions[edge_adjacency[:,0]]
        edge_vecs += tessellation_vecs
        edge_lengths = np.linalg.norm(edge_vecs, axis=1)
        uc_vol = Lattice.calculate_UC_volume(lattice_constants)

        num_uq_nodes = len(np.unique(edge_adjacency))
        
        # features common for all relative densities
        _nodal_ft = torch.ones((num_uq_nodes,1), dtype=torch.float)
        _shifts = torch.tensor(tessellation_vecs, dtype=torch.float)
        _unit_shifts = torch.tensor(unit_shifts, dtype=torch.long)
        _edge_adj = torch.tensor(edge_adjacency.T, dtype=torch.long)
        _nodal_pos = torch.tensor(nodal_positions, dtype=torch.float)

        out_list = []
        assert len(compliance_tensors)>0, f'Lattice {name} does not have enough data'
        avail_reldens = list(compliance_tensors.keys())
        for rel_dens in avail_reldens[reldens_slice]:
            
            if 'fundamental_edge_radii' in lat_data:
                _fund_rel_dens = np.array(list(lat_data['fundamental_edge_radii'].keys()))
                # Find closest relative density but ignore small rounding errors
                _rel_dens = _fund_rel_dens[np.argmin(np.abs(_fund_rel_dens-rel_dens))]
                assert np.abs(_rel_dens-rel_dens)<1e-4, f'Closest relative density {_rel_dens} is not close enough to {rel_dens}'
                edge_radii = np.array(lat_data['fundamental_edge_radii'][_rel_dens]).reshape(-1,1)
                edge_radii = np.concatenate((edge_radii, edge_radii), axis=0)
                assert edge_radii.shape[0]==edge_adjacency.shape[0], f'Edge radii shape {edge_radii.shape} does not match edge adjacency shape {edge_adjacency.shape}'
            else:
                sum_edge_lengths = edge_lengths.sum()
                edge_rad = np.sqrt(rel_dens*uc_vol/(sum_edge_lengths * np.pi))
                edge_radii = edge_rad*np.ones(edge_adjacency.shape[0])

            # ground truth compliance need not be given
            compliance = compliance_tensors[rel_dens]
            if compliance is not None:
                stiffness = np.linalg.inv(compliance) # Mandel
                if graph_ft_format=='Voigt':
                    stiffness = torch.from_numpy(elasticity_func.stiffness_Mandel_to_Voigt(stiffness)).unsqueeze(0)
                    compliance = torch.from_numpy(elasticity_func.compliance_Mandel_to_Voigt(compliance)).unsqueeze(0)
                elif graph_ft_format=='cartesian_4':    
                    compliance = elasticity_func.numpy_Mandel_to_cart_4(compliance)
                    stiffness = elasticity_func.numpy_Mandel_to_cart_4(stiffness)
                    compliance = torch.from_numpy(compliance).unsqueeze(0)
                    stiffness = torch.from_numpy(stiffness).unsqueeze(0)
                elif graph_ft_format=='Mandel':
                    compliance = torch.from_numpy(compliance).unsqueeze(0)
                    stiffness = torch.from_numpy(stiffness).unsqueeze(0)
            else:
                stiffness = None
                compliance = None

            
            edge_ft_list = []
            edge_ft_list_rev = []
            for key in edge_ft_format.split(','):
                if key=='L':
                    edge_ft_list.append(edge_lengths)
                    edge_ft_list_rev.append(edge_lengths)
                elif key=='r':
                    edge_ft_list.append(edge_radii)
                    edge_ft_list_rev.append(edge_radii)
                elif key=='e_vec':
                    edge_unit_vecs = edge_vecs/edge_lengths.reshape(-1,1)
                    edge_ft_list.append(edge_unit_vecs)
                    edge_ft_list_rev.append(-edge_unit_vecs)
                elif key=='euler':
                    v = edge_vecs/edge_lengths.reshape(-1,1)
                    _phi = np.arccos(v[:,2])
                    _th = np.arctan2(v[:,1],v[:,0])+np.pi
                    edge_ft_list.append(np.column_stack(_phi, _th))
                    v = -edge_vecs/edge_lengths.reshape(-1,1)
                    _phi = np.arccos(v[:,2])
                    _th = np.arctan2(v[:,1],v[:,0])+np.pi
                    edge_ft_list_rev.append(np.column_stack(_phi, _th))
                else:
                    raise ValueError(f'Unrecognised edge format string `{key}`')

            edge_features = np.column_stack(edge_ft_list)

            # convert to torch tensors
            _edge_ft = torch.tensor(edge_features, dtype=torch.float)
            
            data = Data(
                # common for all reldens
                name=name,
                positions=_nodal_pos,
                node_attrs=_nodal_ft, 
                edge_index=_edge_adj, 
                shifts=_shifts,
                unit_shifts=_unit_shifts,
                # for this reldens
                edge_attr=_edge_ft, 
                rel_dens=rel_dens,
                stiffness=stiffness,
                compliance=compliance,
                )

            if pre_filter is not None and not pre_filter(data):
                continue
            if pre_transform is not None:
                data = pre_transform(data)
            out_list.append(data)
        return out_list

    def process(self):
        cat = Catalogue.from_file(self.raw_paths[0], 0, regex=self.regex_filter)

        print(f'Processing catalogue {self.catalogue_name}.'
        f' Number of lattices {len(cat)} x {self.nreldens} = {len(cat)*self.nreldens},'
        f' Representation {self.repr}.'
        f' Nodal features: {self.node_ft_format}.'
        f' Edge features: {self.edge_ft_format}.'
        f' Graph feature format {self.graph_ft_format}.'
        )

        if (not self.multiprocessing) or (self.multiprocessing<2):
            print('Running sequential processing...')
            data_list = []
            for lat_data in tqdm(cat):
                data_list.extend(self.process_one(lat_data, self.edge_ft_format, self.graph_ft_format, self.reldens_slice, self.pre_filter, self.pre_transform))
        else:
            raise NotImplementedError('Parallel processing not implemented for now')
            print('Running parallel processing...') # parallel processing is slower! why?
            assert isinstance(self.multiprocessing, int), "multiprocessing has to be boolean or integer"

            with Pool(processes=self.multiprocessing) as p:
                out_data = p.map(self.process_one, cat)

            data_list = [data for out_list in out_data for data in out_list]
            
        if len(data_list)<1:
            raise RuntimeError('Empty data list')
        else:
            torch.save(self.collate(data_list), self.processed_paths[0])