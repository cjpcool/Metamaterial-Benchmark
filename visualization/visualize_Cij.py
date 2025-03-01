"""
Script adopted from HomePY
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def visualizeCij(homogenized_constitutive_matrix, resolution_for_visualization):
    tensor = generate(homogenized_constitutive_matrix)
    """find the E1 in 360 degree x = 0:pi/180:2*pi;"""
    a = np.linspace(0, 2 * np.pi, num=resolution_for_visualization)
    e = np.linspace(-np.pi / 2, np.pi / 2, num=resolution_for_visualization)
    a, e = np.meshgrid(a, e)
    e1 = np.zeros(np.shape(a))
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            """build transformation matrix"""
            trans_z = np.matrix(
                [[np.cos(a[i][j]), -np.sin(a[i][j]), 0.], [np.sin(a[i][j]), np.cos(a[i][j]), 0.], [0., 0., 1.]])
            trans_y = np.matrix(
                [[np.cos(e[i][j]), 0., np.sin(e[i][j])], [0., 1., 0.], [-np.sin(e[i][j]), 0., np.cos(e[i][j])]])
            """calculate the new tensor"""
            N_tensor = transform(tensor, trans_y * trans_z)
            """transform the tensor to 6*6"""
            N_CH = convert_to_matrix(N_tensor)
            """calculate the E1"""
            E = modulus(N_CH)
            e1[i, j] = E[0]
    x, y, z = convert_spherical_coordinates_to_cartesian(a, e, e1)

    V = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    V_normalized = (V - np.min(V)) / (np.max(V) - np.min(V))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8), dpi=150)
    # surf = ax.plot_surface(x, y, z, facecolors=plt.cm.jet(V_normalized), linewidth=0, antialiased=False)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_zlabel('z', fontsize=14)
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(V)
    ax.zaxis.set_major_formatter('{x:.02f}')
    cbar = plt.colorbar(m, ax=ax, pad=0.1)
    cbar.set_label('Value')
    return plt


# def get_visual_results(resolution, voxels, size, homogenized_constitutive_matrix, input_file, use_matplotlib=False):
#
#     number_of_nodes_along_each_axis = resolution + 1
#     resolution_for_visualization = 300
#     voxels = np.reshape(voxels, (resolution * resolution * resolution)).astype(int)
#     element_size = size / resolution
#
#     if use_matplotlib:
#         draw_graph(homogenized_constitutive_matrix, resolution_for_visualization)
#     else:
#         write_effective_youngs_modulus_surface(homogenized_constitutive_matrix, resolution_for_visualization,
#                                                os.path.splitext(input_file)[0] + ".csv")

def transform(itr, tmx):
    ne = itr.size
    nd = itr.ndim

    if ne == 3:
        return

    otr = np.copy(itr)
    otr.fill(0.)
    otr = np.reshape(otr, ne)
    itr_tmp = np.copy(itr)
    itr_tmp = np.reshape(itr_tmp, ne)

    cne = np.cumprod(3 * np.ones(nd)) / 3

    for oe in range(ne):
        ioe = ((np.floor(oe / cne)) % 3).astype(int)
        for ie in range(ne):
            pmx = 1
            iie = ((np.floor(ie / cne)) % 3).astype(int)
            for id1 in range(nd):
                pmx = pmx * tmx[ioe[id1], iie[id1]]
            otr[oe] = otr[oe] + pmx * itr_tmp[ie]

    otr = np.reshape(otr, (3, 3, 3, 3))
    return otr


def change(w):
    """change the index 4 5 6 to 23 31 12"""
    if w < 3:
        a = w
        b = w
    elif w == 3:
        a = 1
        b = 2
    elif w == 4:
        a = 2
        b = 0
    elif w == 5:
        a = 0
        b = 1
    return a, b


def generate(homogenized_constitutive_matrix):
    C = np.full((3, 3, 3, 3), 0.)
    for i in range(6):
        for j in range(6):
            (a, b) = change(i)
            (c, d) = change(j)
            C[a, b, c, d] = homogenized_constitutive_matrix[i, j]
    for i in range(3):
        if i == 2:
            j = 0
        else:
            j = i + 1
        for m in range(3):
            if m == 2:
                n = 0
            else:
                n = m + 1
            C[j, i, n, m] = C[i, j, m, n]
            C[j, i, m, n] = C[i, j, m, n]
            C[i, j, n, m] = C[i, j, m, n]
            C[j, i, m, m] = C[i, j, m, m]
            C[m, m, j, i] = C[m, m, i, j]
    return C


def modulus(homogenized_constitutive_matrix):
    E = np.zeros((6, 1))
    try:
        S = np.linalg.inv(homogenized_constitutive_matrix)
    except np.linalg.LinAlgError:
        return E
    E[0] = 1 / S[0, 0]
    E[1] = 1 / S[1, 1]
    E[2] = 1 / S[2, 2]
    E[3] = 1 / S[3, 3]
    E[4] = 1 / S[4, 4]
    E[5] = 1 / S[5, 5]
    return E


def convert_to_matrix(C):
    CH = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            a, b = change(i)
            c, d = change(j)
            CH[i, j] = C[a, b, c, d]
    return CH


def convert_spherical_coordinates_to_cartesian(azimuth, elevation, r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


def write_effective_youngs_modulus_surface(homogenized_constitutive_matrix, resolution_for_visualization, filename):
    tensor = generate(homogenized_constitutive_matrix)
    """find the E1 in 360 degree x = 0:pi/180:2*pi;"""
    a = np.linspace(0, 2 * np.pi, num=resolution_for_visualization)
    e = np.linspace(-np.pi / 2, np.pi / 2, num=resolution_for_visualization)
    a, e = np.meshgrid(a, e)
    e1 = np.zeros(np.shape(a))
    for i in range(np.shape(a)[0]):
        for j in range(np.shape(a)[1]):
            """build transformation matrix"""
            trans_z = np.matrix(
                [[np.cos(a[i][j]), -np.sin(a[i][j]), 0.], [np.sin(a[i][j]), np.cos(a[i][j]), 0.], [0., 0., 1.]])
            trans_y = np.matrix(
                [[np.cos(e[i][j]), 0., np.sin(e[i][j])], [0., 1., 0.], [-np.sin(e[i][j]), 0., np.cos(e[i][j])]])
            """calculate the new tensor"""
            N_tensor = transform(tensor, trans_y * trans_z)
            """transform the tensor to 6*6"""
            N_CH = convert_to_matrix(N_tensor)
            """calculate the E1"""
            E = modulus(N_CH)
            e1[i, j] = E[0]
    x, y, z = convert_spherical_coordinates_to_cartesian(a, e, e1)
    x_max = np.max(x)
    y_max = np.max(y)
    z_max = np.max(z)
    c = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    f = open(filename, "w")
    f.write("X,Y,Z,C\n")
    for i in range((np.shape(x)[0])):
        for j in range((np.shape(x)[1])):
            f.write(str(.5 * (x[i][j] / x_max + 1.)) + "," + str(.5 * (y[i][j] / y_max + 1.)) + "," + str(
                .5 * (z[i][j] / z_max + 1.)) + "," + str(c[i][j]) + "\n")
    f.close()





# # Example usage
# CH = np.array([[2.06421624e+02, 8.36966457e+01, 8.36973893e+01, 4.30362269e-05, -2.71565216e-06, 1.12651019e-04],
#                [8.36966457e+01, 2.06420208e+02, 8.36965263e+01, 1.32979100e-05, -3.33010012e-07, 3.22262982e-05],
#                [8.36973893e+01, 8.36965263e+01, 2.06422091e+02, 2.46720977e-05, 1.00946670e-05, 9.20790735e-05],
#                [4.30362269e-05, 1.32979100e-05, 2.46720977e-05, 6.29092353e+01, 1.06677403e-05, 1.79537872e-05],
#                [-2.71565217e-06, -3.33010013e-07, 1.00946670e-05, 1.06677403e-05, 6.29091795e+01, 2.33608608e-05],
#                [1.12651019e-04, 3.22262982e-05, 9.20790735e-05, 1.79537872e-05, 2.33608608e-05, 6.29094788e+01]])

# draw_graph(CH, 50)
