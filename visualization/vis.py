import numpy as np
from matplotlib.cm import ScalarMappable
from torch_cluster import radius, radius_graph

from utils.lattice_utils import plot_lattice
import os
from utils.mat_utils import frac_to_cart_coords, get_pbc_cutoff_graphs
import torch
from matplotlib import pyplot as plt
import pyvista
import seaborn as sns
from matplotlib import cm

def plot_origin_lattice_from_path(path, name, cutoff=1.0,max_num_neighbors_threshold=3, save_dir=None):
    full_path = os.path.join(path,name)
    lattice_npz = np.load(full_path)
    frac_coords = lattice_npz['origin_frac_coords']
    lengths = lattice_npz['origin_lengths']
    angles = lattice_npz['origin_angles']
    atom_types = lattice_npz['origin_atom_types']
    num_atoms = torch.tensor([atom_types.shape[0]])
    frac_coords, lengths, angles = torch.from_numpy(frac_coords),torch.from_numpy(lengths), torch.from_numpy(angles)
    cart_coords = frac_to_cart_coords(frac_coords,lengths,
                                      angles, num_atoms)
    print('num_atoms', num_atoms)
    try:
        # raise Exception
        edge_index = lattice_npz['origin_edge_index']
    except:
        edge_index, _,_ = get_pbc_cutoff_graphs(cart_coords, lengths, angles, num_atoms, cutoff=cutoff, max_num_neighbors_threshold=max_num_neighbors_threshold)
    # print('edge_index \n', edge_index)
    plot_lattice(cart_coords,edge_index.T, save_dir=save_dir)


def plot_lattice_from_path(path, name, cutoff=2.0,max_num_neighbors_threshold=5, save_dir=None, plot_method='pyvista' ):
    full_path = os.path.join(path,name)
    lattice_npz = np.load(full_path)
    frac_coords = lattice_npz['frac_coords']
    lengths = lattice_npz['lengths']
    angles = lattice_npz['angles']
    atom_types = lattice_npz['atom_types']
    num_atoms = torch.tensor([atom_types.shape[0]])
    frac_coords, lengths, angles = torch.from_numpy(frac_coords),torch.from_numpy(lengths).unsqueeze(0), torch.from_numpy(angles).unsqueeze(0)
    cart_coords = frac_to_cart_coords(frac_coords, lengths,
                                      angles, num_atoms)
    # cart_coords = frac_coords

    print('num_atoms', num_atoms)
    try:
        # raise Exception
        edge_index = lattice_npz['edge_index']
    except:
        edge_index, _,_ = radius_graph(cart_coords, cutoff, max_num_neighbors=max_num_neighbors_threshold)

    if plot_method == 'pyvista':
        visualizeLattice_interactive(frac_coords, edge_index.T, file_name=save_dir)
    elif plot_method =='1':
        visualizeLattice(frac_coords, edge_index.T, save_dir=save_dir)
    else:
        plot_lattice(frac_coords,edge_index.T, save_dir=save_dir)


def visualizeLattice(nodes, struts, save_dir=None, dpi=150):
    """
    Visualize the lattice structure from the specified file.

    Parameters:
        save_dir (str): Path to the file containing node and strut data.
        dpi (int): Dots per inch setting for the plot resolution.
        (dpi represents dots per inch, number can be adjusted based on the need)
    """

    # Initialize containers for nodes and struts
    struts = struts.T
    # Plot the lattice structure
    fig = plt.figure(dpi=dpi, figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the title using the file name

    # ax.set_title()

    # Customize background color
    ax.set_facecolor((1, 1, 1))  # Light gray background
    ax.grid(True)

    # Plot nodes with a solid color (e.g., yellow with black edge)
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='yellow', edgecolor='black', s=30)

    # Plot struts with a solid color (e.g., blue)
    for strut in struts:
        start_node = nodes[strut[0], :]
        end_node = nodes[strut[1], :]
        ax.plot([start_node[0], end_node[0]],
                [start_node[1], end_node[1]],
                [start_node[2], end_node[2]], 'b-', linewidth=1)

    # set the elevation (elev) and azimuth (azim) angles of the plot
    ax.view_init(elev=10, azim=30)  # these numbers can be adjust to shown figures with different viewing perspective

    # # Turn off the grid
    # ax.grid(False)
    if save_dir is not None:
        plt.savefig(save_dir,bbox_inches='tight', dpi=300)
    else:
        plt.show()


def visualizeLattice_interactive(nodes, edges, file_name=None):
    """
    Visualize the lattice structure from the specified file interactively.

    Parameters:
    """

    # Initialize containers for nodes and struts
    edges = edges.T

    # We must "pad" the edges to indicate to vtk how many points per edge
    padding = np.empty(edges.shape[0], int) * 2
    padding[:] = 2
    edges_w_padding = np.vstack((padding, edges.T)).T

    mesh = pyvista.PolyData(nodes, edges_w_padding)
    colors = range(edges.shape[0])

    if file_name is None:
        mesh.plot(
            scalars=colors,
            render_lines_as_tubes=True,
            style='wireframe',
            line_width=10,
            cmap='jet',
            show_scalar_bar=False,
            background='w',
            color='lightblue',
        )
    else:
        plotter = pyvista.Plotter(notebook=False, off_screen=True)
        mesh.plot(
            scalars=colors,
            render_lines_as_tubes=True,
            style='wireframe',
            line_width=10,
            cmap='jet',
            show_scalar_bar=False,
            background='w',
            color='lightblue',
        )

        plotter.open_gif(file_name)

        plotter.show(auto_close=False)
        plotter.write_frame()

        plotter.close()

from matplotlib.cm import ScalarMappable

def plot_ellipsoid_colormap_modulus(young_modulus, save_path, property_name):
    if len(young_modulus) != 3:
        raise ValueError(f"{property_name} Must contain three values [Ex, Ey, Ez].")

    Ex, Ey, Ez = young_modulus

    u = np.linspace(0, np.pi, 50)
    v = np.linspace(0, 2 * np.pi, 50)
    u, v = np.meshgrid(u, v)

    X = Ex * np.sin(u) * np.cos(v)
    Y = Ey * np.sin(u) * np.sin(v)
    Z = Ez * np.cos(u)

    R = np.sqrt((X / Ex)**2 + (Y / Ey)**2 + (Z / Ez)**2)
    R_normalized = (R - R.min()) / (R.max() - R.min())

    colors = plt.cm.jet(R_normalized)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X, Y, Z,
        rstride=1, cstride=1,
        facecolors=colors,
        linewidth=0,
        antialiased=True
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"3D Ellipsoid of {property_name}", fontsize=20)

    mappable = ScalarMappable(cmap='jet')
    mappable.set_array(R)

    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label("Normalized Radius R")

    plt.savefig(save_path, dpi=300)
    plt.close(fig)

from matplotlib.colors import Normalize
def plot_directional_modulus_from_poissons(nu, save_path=None, E_default=1.0, cmap='jet'):
    # nu: 6 values [ν12, ν13, ν23, ν21, ν31, ν32]
    # Assume E1 = E2 = E3 = E_default
    if isinstance(nu, list):
        nu = np.array(nu)
    E1 = E2 = E3 = E_default
    nu12, nu13, nu23, nu21, nu31, nu32 = nu
    # Build the compliance matrix S
    S = np.array([
        [1 / E1, -nu12 / E1, -nu13 / E1],
        [-nu21 / E2, 1 / E2, -nu23 / E2],
        [-nu31 / E3, -nu32 / E3, 1 / E3]
    ])

    # Create spherical grid
    n_u, n_v = 100, 100
    u = np.linspace(0, np.pi, n_u)
    v = np.linspace(0, 2 * np.pi, n_v)
    U, V = np.meshgrid(u, v)
    E_dir = np.zeros_like(U)

    # Compute directional modulus: E(n) = 1/(nᵀ S n)
    for i in range(n_u):
        for j in range(n_v):
            theta = U[i, j]
            phi = V[i, j]
            n_vec = np.array([np.sin(theta) * np.cos(phi),
                              np.sin(theta) * np.sin(phi),
                              np.cos(theta)])
            E_dir[i, j] = 1.0 / (n_vec.T @ S @ n_vec)

    # Convert spherical to Cartesian coordinates for plotting
    X = E_dir * np.sin(U) * np.cos(V)
    Y = E_dir * np.sin(U) * np.sin(V)
    Z = E_dir * np.cos(U)

    # Plot the modulus surface
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    norm = Normalize(vmin=E_dir.min(), vmax=E_dir.max())
    colors = plt.get_cmap(cmap)(norm(E_dir))

    ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0, antialiased=True)
    ax.set_title("Directional Modulus Surface (E=1.0) for Poisson's Ratio")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Add colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, aspect=15, pad=0.1)
    cbar.set_label("Normalized Modulus")

    plt.savefig(save_path, dpi=300)
    # plt.show()
    plt.close(fig)


if __name__ == "__main__":
    poisson_matrix = np.array([0.25, 0.20, 0.15, 0.20, 0.30, 0.10])

    plot_directional_modulus_from_poissons(poisson_matrix, )