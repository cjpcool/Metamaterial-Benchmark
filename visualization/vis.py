import numpy as np
from torch_cluster import radius, radius_graph

from utils.lattice_utils import plot_lattice
import os
from utils.mat_utils import frac_to_cart_coords, get_pbc_cutoff_graphs
import torch
from matplotlib import pyplot as plt
import pyvista

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

    # Plot the lattice structure
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the title using the file name

    ax.set_title(save_dir)

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
        plt.savefig(save_dir)
    plt.show()


def visualizeLattice_interactive(nodes, edges, file_name=None):
    """
    Visualize the lattice structure from the specified file interactively.

    Parameters:
    """

    # Initialize containers for nodes and struts
    edges = edges

    # We must "pad" the edges to indicate to vtk how many points per edge
    padding = np.empty(edges.shape[0], int) * 2
    padding[:] = 2
    edges_w_padding = np.vstack((padding, edges.T)).T

    if file_name is None:
        mesh = pyvista.PolyData(nodes, edges_w_padding)

        colors = range(edges.shape[0])
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
        mesh = pyvista.PolyData(nodes, edges_w_padding)

        plotter = pyvista.Plotter(notebook=False, off_screen=True)

        colors = range(edges.shape[0])
        plotter.add_mesh(mesh, scalars=colors, line_width=10, cmap='jet',
                         )

        plotter.add_mesh(mesh, scalars=colors, render_lines_as_tubes=True, style='wireframe', line_width=10, cmap='jet',
                         name='mesh')

        # 保存为 GIF
        plotter.show(auto_close=False)  # 不自动关闭
        plotter.export_gif(file_name)  # 导出为 GIF
        plotter.close()




if __name__ == '__main__':

    path = './generated_mat/lattices/lattices'
    file_names = os.listdir(path)
    save_path = './vis/generated_mat/test'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for file_name in file_names:
        save_dir = os.path.join(save_path,file_name[:-3]+'png')
        plot_lattice_from_path(path, file_name, save_dir=save_dir, plot_method = '1')

    # plot_origin_lattice_from_path(path, name,cutoff=1., max_num_neighbors_threshold=5)
