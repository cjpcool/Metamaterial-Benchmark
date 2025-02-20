import numpy as np
import pickle
from typing import List, Dict, Tuple
import os
from pathlib import Path
import argparse

def parse_cif_file(file_path: str) -> Dict:
    """Parse a single .cif file and return its parameters."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    structure = {
        'lengths': [],
        'angles': [],
        'coordinates': [],
        'edge_index': [],
        'youngs_modulus': [],
        'shear_modulus': [],
        'poisson_ratio': []
    }
    
    # Parse lattice parameters and mechanical properties
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        
        # Parse mechanical properties
        if line.startswith('_youngs_modulus'):
            current_section = 'youngs'
            continue
        elif line.startswith('_shear_modulus'):
            current_section = 'shear'
            continue
        elif line.startswith('_poisson_ratio'):
            current_section = 'poisson'
            continue
        elif line.startswith('_params'):
            current_section = None
            continue
            
        # Parse values based on their prefixes
        if current_section == 'youngs' and (line.startswith('_x') or line.startswith('_y') or line.startswith('_z')):
            try:
                structure['youngs_modulus'].append(float(parts[-1]))
            except (ValueError, IndexError):
                continue
                
        elif current_section == 'shear' and (line.startswith('_xy') or line.startswith('_xz') or line.startswith('_yz')):
            try:
                structure['shear_modulus'].append(float(parts[-1]))
            except (ValueError, IndexError):
                continue
                
        elif current_section == 'poisson' and (line.startswith('_xy') or line.startswith('_xz') or 
                                             line.startswith('_yx') or line.startswith('_yz') or
                                             line.startswith('_zx') or line.startswith('_zy')):
            try:
                structure['poisson_ratio'].append(float(parts[-1]))
            except (ValueError, IndexError):
                continue
                
        # Parse lattice parameters
        elif line.startswith('_cell_length'):
            try:
                structure['lengths'].append(float(parts[-1]))
            except (ValueError, IndexError):
                continue
        elif line.startswith('_cell_angle'):
            try:
                structure['angles'].append(float(parts[-1]))
            except (ValueError, IndexError):
                continue

    # Parse atomic coordinates
    for line in lines:
        line = line.strip()
        if line.startswith('atom_'):
            try:
                parts = line.split()
                coords = [float(parts[1]), float(parts[2]), float(parts[3])]
                structure['coordinates'].append(coords)
            except (ValueError, IndexError):
                continue

    # Parse edge indices
    edge_section = False
    for line in lines:
        line = line.strip()
        if '_edge_from_atom' in line:
            edge_section = True
            continue
        if edge_section and line and not line.startswith('_'):
            try:
                parts = line.split()
                if len(parts) == 2:
                    from_atom, to_atom = map(int, parts)
                    structure['edge_index'].append([from_atom, to_atom])
            except (ValueError, IndexError):
                continue

    # Convert to numpy arrays
    structure['lengths'] = np.array(structure['lengths'])
    structure['angles'] = np.array(structure['angles'])
    structure['coordinates'] = np.array(structure['coordinates'], dtype=np.float64)
    structure['edge_index'] = np.array(structure['edge_index'])
    structure['youngs_modulus'] = np.array(structure['youngs_modulus'])
    structure['shear_modulus'] = np.array(structure['shear_modulus'])
    structure['poisson_ratio'] = np.array(structure['poisson_ratio'])
    structure['mech_props'] = np.concatenate([
        structure['youngs_modulus'],
        structure['shear_modulus'],
        structure['poisson_ratio']
    ])
    # Debug print
    print(f"File: {file_path}")
    print(f"Young's modulus: {len(structure['youngs_modulus'])}")
    print(f"Shear modulus: {len(structure['shear_modulus'])}")
    print(f"Poisson ratio: {len(structure['poisson_ratio'])}")
    
    # Verify we have the correct number of parameters
    if len(structure['lengths']) != 3 or len(structure['angles']) != 3:
        raise ValueError("Missing lattice parameters")
    if (len(structure['youngs_modulus']) != 3 or 
        len(structure['shear_modulus']) != 3 or 
        len(structure['poisson_ratio']) != 6):
        raise ValueError("Missing mechanical properties")
    
    return structure

def process_directory(directory_path: str, output_file: str):
    """Process all .cif files in the directory and save to a pickle file."""
    structures = []
    
    # Get all .cif files in the directory
    cif_files = list(Path(directory_path).glob('*.cif'))
    print("length of cif files: ", len(cif_files))
    print(f"Found {len(cif_files)} .cif files")
    
    # Process each file
    for file_path in cif_files:
        try:
            structure = parse_cif_file(str(file_path))
            structures.append(structure)
            print(f"Processed {file_path.name}")
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
    
    # Save to pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(structures, f)
    
    print(f"\nSaved {len(structures)} structures to {output_file}")

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process CIF files and save structures.')
    parser.add_argument('--cond', action='store_true', help='Use conditional generation output path')
    args = parser.parse_args()

    # Directory containing .cif files
      # Change this to your directory path

    if args.cond:
        print("Using conditional generation output path")
        directory = "/home/zhang/Metamaterial-Benchmark/CrystaLLM/cond_gen_lattices"
        output_file = "CrystaLLM/cond_gen_lattices.pkl"
    else:
        print("Using unconditional generation output path")
        directory = "/home/zhang/Metamaterial-Benchmark/CrystaLLM/uncond_gen_lattices"
        output_file = "CrystaLLM/uncond_gen_lattices.pkl"
    
    process_directory(directory, output_file)