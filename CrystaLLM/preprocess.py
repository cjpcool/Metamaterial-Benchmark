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
_params
_cell_length_a   {lengths[0]:.6f}
_cell_length_b   {lengths[1]:.6f}
_cell_length_c   {lengths[2]:.6f}
_cell_angle_alpha   {angles[0]:.6f}
_cell_angle_beta    {angles[1]:.6f}
_cell_angle_gamma   {angles[2]:.6f}"""

    cif_content += "\nloop_\n_atom_index\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
    for i, coord in enumerate(frac_coords):
        cif_content += f"atom_{i}  {coord[0]:.3f}  {coord[1]:.3f}  {coord[2]:.3f}\n"
    
    cif_content += "\nloop_\n_edge_from_atom\n_edge_to_atom\n"
    for i in range(len(edge_index)):
        from_atom = edge_index[i][0]
        to_atom = edge_index[i][1]
        cif_content += f"{from_atom}  {to_atom}\n"
    cif_content += "\n\n"
    
    return cif_content

if __name__ == "__main__":
    # Number of augmentation rounds
    n_augment_rounds = 20

    # Load the lattices
    lattices = pickle.load(open("/home/zhang/Metamaterial-Benchmark/unit_cell_catalog/train_set.pkl", "rb"))

    cif_data = []
    structure_count = 0

    for round_idx in range(n_augment_rounds):
        for i, lattice in enumerate(lattices):
            # Copy the lattice to avoid in-place modifications
            lattice_copy = dict(lattice)
            
            frac_coords = lattice_copy['frac_coords']
            edge_index = lattice_copy['edge_index'].T
            
            # Shuffle the atom order
            num_atoms = len(frac_coords)
            permutation = np.random.permutation(num_atoms)
            new_frac_coords = frac_coords[permutation]
            
            # Map old indices to new indices
            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(permutation)}
            
            # Update edges
            new_edge_index = []
            for edge in edge_index:
                old_from, old_to = edge
                new_from = old_to_new[old_from]
                new_to = old_to_new[old_to]
                new_edge_index.append([new_from, new_to])
            new_edge_index = np.array(new_edge_index, dtype=int)
            
            # Optionally shuffle the edges as well
            edge_perm = np.random.permutation(len(new_edge_index))
            new_edge_index = new_edge_index[edge_perm]
            
            # Store the new data back
            lattice_copy['frac_coords'] = new_frac_coords
            lattice_copy['edge_index'] = new_edge_index.T
            
            cif_content = create_cif_content(lattice_copy, f"{round_idx}_{structure_count}")
            cif_data.append(cif_content)
            structure_count += 1
    print("First structure:")
    print(cif_data[0])
    # Save as compressed pickle file
    with gzip.open('/home/zhang/Metamaterial-Benchmark/CrystaLLM/aug_train_structures.pkl.gz', 'wb') as f:
        pickle.dump(cif_data, f)

    print(f"Saved {len(cif_data)} structures to /home/zhang/Metamaterial-Benchmark/CrystaLLM/aug_train_structures.pkl.gz")