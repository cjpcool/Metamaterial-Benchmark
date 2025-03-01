import numpy as np
from matplotlib.cm import ScalarMappable
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
        plt.savefig(save_dir,bbox_inches='tight')
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
        # 直接交互式显示
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
        # 使用离屏绘图模式
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

        # 开启 GIF 输出，并指定保存的文件名
        plotter.open_gif(file_name)

        # 如果你希望捕获多帧动画，可以在此处调整摄像机或其他属性，并多次调用 write_frame()
        # 例如，简单捕获当前帧：
        plotter.show(auto_close=False)
        plotter.write_frame()

        # 关闭绘图窗口，同时完成 GIF 的保存
        plotter.close()

from matplotlib.cm import ScalarMappable

def plot_ellipsoid_colormap(young_modulus, save_path):
    if len(young_modulus) != 3:
        raise ValueError("young_modulus Must contain three values [Ex, Ey, Ez].")

    Ex, Ey, Ez = young_modulus

    # 1) 生成椭球网格
    u = np.linspace(0, np.pi, 50)
    v = np.linspace(0, 2 * np.pi, 50)
    u, v = np.meshgrid(u, v)

    X = Ex * np.sin(u) * np.cos(v)
    Y = Ey * np.sin(u) * np.sin(v)
    Z = Ez * np.cos(u)

    # 2) 定义用于控制颜色的标量场
    R = np.sqrt((X / Ex)**2 + (Y / Ey)**2 + (Z / Ez)**2)
    R_normalized = (R - R.min()) / (R.max() - R.min())

    # 3) 将标量 R 映射为 RGBA 颜色
    colors = plt.cm.jet(R_normalized)

    # 4) 绘制 3D 表面
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X, Y, Z,
        rstride=1, cstride=1,
        facecolors=colors,  # 指定颜色
        linewidth=0,
        antialiased=True
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Ellipsoid of Young's Modulus")

    # 5) 创建一个与颜色对应的 ScalarMappable，并关联到同一个 cmap
    mappable = ScalarMappable(cmap='jet')
    # 这里设置要显示在 colorbar 上的原始数据 (R)，而不是 R_normalized
    mappable.set_array(R)

    # 6) 在同一个 Axes (ax) 上放置 colorbar
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label("Normalized Radius R")

    # 7) 保存并关闭
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# ========== 示例调用 ==========
if __name__ == "__main__":
    moduli = [3.0, 2.0, 4.0]
    plot_ellipsoid_colormap(moduli, "ellipsoid_colormap.png")
