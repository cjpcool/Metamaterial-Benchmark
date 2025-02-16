import numpy as np
import matplotlib.pyplot as plt


def visualizeVox(voxel, save_fig=None):
    # Generate binary voxel data
    voxel_data = voxel

    # Create a 3D plot
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    # Get the coordinates of the voxels
    dimensions = voxel.shape
    x, y, z = np.indices(dimensions)

    # Plot the voxels
    ax.voxels(voxel_data)

    # set the elevation (elev) and azimuth (azim) angles of the plot
    ax.view_init(elev=25, azim=30)  # these numbers can be adjust to shown figures with different viewing perspective

    # Set labels
    ax.set_xlabel('voxel X')
    ax.set_ylabel('voxel Y')
    ax.set_zlabel('voxel Z')

    # Turn off the grid
    # ax.grid(False)

    # Show the plot
    if save_fig is not None:
        plt.savefig(save_fig)
    else:
        plt.show()

def generate_voxel(n, node, strut, radius):
    """
    Generate a voxel grid and calculate the relative density.

    Parameters:
        n (int): Number of voxels along each axis.
        address (str): File location of the wireframe.
        radius (float): Radius for determining active voxels.

    Returns:
        tuple: Voxel grid (3D numpy array) and density (float).
    """
    strut = strut.T
    size = 1.0 / n  # initial size of voxels
    voxel = np.zeros((n, n, n))  # initial grid with zeros
    # Generate a list of centers of voxel
    voxel_c = np.zeros((n ** 3, 6))
    p = 0  # p count the number of all voxels
    for i in range(1, n + 1):  # i for z axis
        for j in range(1, n + 1):  # j for y axis
            for k in range(1, n + 1):  # k for x axis
                p += 1
                voxel_c[p - 1, 0:3] = [k, j, i]  # save index along x,y,z axis
                # save coordinate along x,y,z axis
                voxel_c[p - 1, 3:6] = [(k - 0.5) * size, (j - 0.5) * size, (i - 0.5) * size]

                # Get the voxel close to the strut within a certain distance
    # node, strut = read_strut(address)  # get the information of strut
    for i in range(len(voxel_c)):  # for each voxel, decide if it is active
        center = voxel_c[i, 3:6]  # voxel center position
        for j in range(len(strut)):  # for each strut, get the distance to the voxel
            start_n = node[strut[j, 0] - 1, :]  # start node coordinate
            end_n = node[strut[j, 1] - 1, :]  # end node coordinate

            # determine if alpha and beta are acute angles
            alpha = np.degrees(np.arccos(np.dot((center - start_n), (end_n - start_n)) /
                                         (np.linalg.norm(center - start_n) * np.linalg.norm(end_n - start_n))))
            beta = np.degrees(np.arccos(np.dot((center - end_n), (start_n - end_n)) /
                                        (np.linalg.norm(center - end_n) * np.linalg.norm(start_n - end_n))))

            if alpha < 90 and beta < 90:  # if not acute angle, distance to line
                distance = np.linalg.norm(np.cross(end_n - start_n, center - start_n)) / np.linalg.norm(
                    end_n - start_n)
            else:  # if it is acute angle, distance to node
                distance = min(np.linalg.norm(center - start_n), np.linalg.norm(center - end_n))

            if distance <= radius:  # if distance less than radius, activate it
                voxel[int(voxel_c[i, 0]) - 1, int(voxel_c[i, 1]) - 1, int(voxel_c[i, 2]) - 1] = 1
                break  # move to the next voxel

    density = np.sum(voxel) / n ** 3  # calculate the relative density
    return voxel, density
