import pickle
import os
import gzip
import numpy as np
import random
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def create_cif_content(lattice, index):
    """Convert a lattice dictionary into a CIF file format."""
    
    # Extract the lattice parameters
    lengths = lattice['lengths'][0]  
    angles = lattice['angles'][0]    
    frac_coords = lattice['frac_coords']
    edge_index = lattice['edge_index'].T
    
    y = lattice['mech_props'][0]
    youngs_modulus = y[0:3]
    shear_modulus = y[3:6]
    poisson_ratio = y[6:12]
    
    cif_content = f"""lattice_data_

_mechanical
_youngs_modulus
_x     {youngs_modulus[0]:.6f}
_y     {youngs_modulus[1]:.6f}
_z     {youngs_modulus[2]:.6f}
_shear_modulus
_xy     {shear_modulus[0]:.6f}
_xz     {shear_modulus[1]:.6f}
_yz     {shear_modulus[2]:.6f}
_poisson_ratio
_xy     {poisson_ratio[0]:.6f}
_xz     {poisson_ratio[1]:.6f}
_yx     {poisson_ratio[2]:.6f}
_yz     {poisson_ratio[3]:.6f}
_zx     {poisson_ratio[4]:.6f}
_zy     {poisson_ratio[5]:.6f}
"""
    
    return cif_content

if __name__ == "__main__":
    # Number of augmentation rounds
    n_augment_rounds = 20

    # Load the lattices
    lattices = pickle.load(open("/home/zhang/Metamaterial-Benchmark/unit_cell_catalog/test_set.pkl", "rb"))

    cif_data = []
    structure_count = 0

    
    for i, lattice in enumerate(lattices):
        # Copy the lattice to avoid in-place modifications
        
        cif_content = create_cif_content(lattice, i)
        cif_data.append(cif_content)
        structure_count += 1

    # Save as compressed pickle file
    with gzip.open('/home/zhang/Metamaterial-Benchmark/CrystaLLM/test_structures.pkl.gz', 'wb') as f:
        pickle.dump(cif_data, f)

    print(f"Saved {len(cif_data)} structures to /home/zhang/Metamaterial-Benchmark/CrystaLLM/test_structures.pkl.gz")