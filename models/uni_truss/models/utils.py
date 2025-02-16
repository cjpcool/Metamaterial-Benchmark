import torch
import numpy as np
import pandas as pd

s_name = ['E11', 'E22', 'E33', 'G23', 'G31', 'G12', 'v21', 'v31', 'v32', 'v12', 'v13', 'v23']
c_name = ['C11', 'C12', 'C13', 'C22', 'C23', 'C33', 'C44', 'C55', 'C66']

numNodes = 27
dim = 3

def vec2tensor(ex):
    exC = np.zeros([6,6])
    exC[0,0] = ex[0]; exC[0,1] = ex[1]; exC[0,2] = ex[2]
    exC[1,0] = ex[1]; exC[1,1] = ex[3]; exC[1,2] = ex[4]; 
    exC[2,0] = ex[2]; exC[2,1] = ex[4]; exC[2,2] = ex[5]
    exC[3,3] = ex[6]; exC[4,4] = ex[7]; exC[5,5] = ex[8]
    return exC

def torch_vec2tensor(c_vec, length):
    c_tensor = torch.zeros([length,6,6])
    exC = torch.zeros([6,6])
    for i in range(length):
        ex = c_vec[i,:]
        exC[0,0] = ex[0]; exC[0,1] = ex[1]; exC[0,2] = ex[2]
        exC[1,0] = ex[1]; exC[1,1] = ex[3]; exC[1,2] = ex[4]; 
        exC[2,0] = ex[2]; exC[2,1] = ex[4]; exC[2,2] = ex[5]
        exC[3,3] = ex[6]; exC[4,4] = ex[7]; exC[5,5] = ex[8]
        c_tensor[i,:,:] = exC
    return c_tensor

def torch_tensor2s(c_tensor):
    s_vec = torch.ones([c_tensor.shape[0], len(s_name)])*1e-6
    for i in range(c_tensor.shape[0]):
        c_ = c_tensor[i,:,:]
        if torch.det(c_) == 0.:
            pass
        else:
            s_tensor = torch.inverse(c_)
            s_vec_ = torch_s2vec(s_tensor)
            s_vec[i,:] = s_vec_
    return s_vec

def torch_s2vec(s_matrix):
    s_vec = torch.ones([12])
    s_vec[0] = 1./s_matrix[0,0]; s_vec[1] = 1./s_matrix[1,1]; s_vec[2] = 1./s_matrix[2,2]
    s_vec[3] = 1./s_matrix[3,3]; s_vec[4] = 1./s_matrix[4,4]; s_vec[5] = 1./s_matrix[5,5]
    s_vec[6] = -s_matrix[0,1]*1./s_matrix[1,1]
    s_vec[7] = -s_matrix[0,2]*1./s_matrix[2,2]
    s_vec[8] = -s_matrix[1,2]*1./s_matrix[2,2]
    s_vec[9] = -s_matrix[1,0]*1./s_matrix[0,0]
    s_vec[10] = -s_matrix[2,0]*1./s_matrix[0,0]
    s_vec[11] = -s_matrix[2,1]*1./s_matrix[1,1]
    return s_vec[:]

def tensor2vec(exC):
    ex = np.zeros([9])
    ex[0] = exC[0,0]; 
    ex[1] = exC[0,1]; 
    ex[2] = exC[0,2]; 
    ex[3] = exC[1,1]; ex[4] = exC[1,2]; ex[5] = exC[2,2]; 
    ex[6] = exC[3,3]; ex[7] = exC[4,4]; ex[8] = exC[5,5]; 
    return ex

def s2vec(s_matrix):
    s_vec = np.ones([12])
    s_vec[0] = 1./s_matrix[0,0]; s_vec[1] = 1./s_matrix[1,1]; s_vec[2] = 1./s_matrix[2,2]
    s_vec[3] = 1./s_matrix[3,3]; s_vec[4] = 1./s_matrix[4,4]; s_vec[5] = 1./s_matrix[5,5]
    s_vec[6] = -s_matrix[0,1]*s_vec[1]
    s_vec[7] = -s_matrix[0,2]*s_vec[2]
    s_vec[8] = -s_matrix[1,2]*s_vec[2]
    s_vec[9] = -s_matrix[1,0]*s_vec[0]
    s_vec[10] = -s_matrix[2,0]*s_vec[0]
    s_vec[11] = -s_matrix[2,1]*s_vec[1]
    return s_vec

def slerp(val, low, high):
    low = torch.unsqueeze(low, dim=0)
    high = torch.unsqueeze(high, dim=0)
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).sum()

def stiffness_weighted_loss(c_pred, c):
    paramDim = c.shape[-1]
    c_mse = 0.
    weight = 1.
    for i in range(paramDim):
        if (i == 1) | (i ==2) | (i == 4):
            weight = 80.
        elif (i == 6) | (i == 7)| (i == 8):
            weight = 30.
        else:
            weight = 15.
        c_mse += weighted_mse_loss(c_pred[:,i], c[:,i], weight)

    return c_mse
    
def writeConn(adj, nodes):
    mirror = 1
    
    beam1 = []; beam2 = []
    beam1_1 = []; beam2_1 = []
    beam1_2 = [];beam2_2 = []

    A_plot = adj.copy()
    A_plot[np.triu_indices(A_plot.shape[0])] = 0.
    row = np.nonzero(A_plot)[0]
    col = np.nonzero(A_plot)[1]


    for i,j in zip(row,col):
        beam1_1.append(nodes[i,:])
        beam2_1.append(nodes[j,:])

    if mirror == 1:
        for i,j in zip(beam1_1, beam2_1):
        #     -y
            temp1 = i.copy();temp2 = j.copy()
            temp1[1] = -i[1];temp2[1] = -j[1]
            beam1_2.append(temp1);beam2_2.append(temp2)
        #     -x & -y
            temp1 = i.copy();temp2 = j.copy()
            temp1[0] = -i[0];temp2[0] = -j[0]
            temp1[1] = -i[1];temp2[1] = -j[1]
            beam1_2.append(temp1);beam2_2.append(temp2)

        #     -x
            temp1 = i.copy();temp2 = j.copy()
            temp1[0] = -i[0];temp2[0] = -j[0]
            beam1_2.append(temp1);beam2_2.append(temp2)

        beam1 = np.concatenate((np.array(beam1_1),np.array(beam1_2)),axis = 0)
        beam2 = np.concatenate((np.array(beam2_1),np.array(beam2_2)),axis = 0)
    else:
        beam1 = np.array(beam1_1)
        beam2 = np.array(beam2_1)
    for i in range(beam1.shape[0]):
        temp1 = beam1[i,:].copy().reshape((1,3));temp2 = beam2[i,:].copy().reshape((1,3))
        temp1[:,-1] =  -beam1[i][-1];temp2[:,-1] = -beam2[i][-1]
        beam1 = np.concatenate((beam1,temp1),axis = 0);beam2 = np.concatenate((beam2,temp2),axis = 0);
    all_nodes = np.concatenate((beam1, beam2))
    uni_nodes = np.unique(all_nodes, axis =0)
    conn_list = []
    for i in range(beam1.shape[0]):
        n1 = beam1[i]
        n2 = beam2[i]
        idx1 = np.where(np.sum(uni_nodes == n1, axis = 1)==3)[0]
        idx2 = np.where(np.sum(uni_nodes == n2, axis = 1)==3)[0]
        conn = np.array([idx1,idx2]).squeeze().astype(int)
        conn_list.append(conn)
    conn_list = np.array(conn_list).astype(int)
    conn_list = np.unique(conn_list, axis = 0)
    return uni_nodes, conn_list

nodesInit = np.array([
    [0.,0.,0.],
    [0.,0.5,0.],
    [0.,1.,0.],
    [0.5,1.,0.],
    [0.5,0.5,0.],
    [0.5,0.,0.],
    [1.,0.,0.],
    [1.,0.5,0.],
    [1.,1.,0.],
    [0.,0.,0.5],
    [0.,0.5,0.5],
    [0.,1.,0.5],
    [0.5,1.,0.5],
    [0.5,0.5,0.5],
    [0.5,0.,0.5],
    [1.,0.,0.5],
    [1.,0.5,0.5],
    [1.,1.,0.5],
    [0.,0.,1.],
    [0.,0.5,1.],
    [0.,1.,1.],
    [0.5,1.,1.],
    [0.5,0.5,1.],
    [0.5,0.,1.],
    [1.,0.,1.],
    [1.,0.5,1.],
    [1.,1.,1.]])

if numNodes == 27:
    x_edge_nodes = [3, 5, 21, 23]
    y_edge_nodes = [1, 7, 19, 25]
    z_edge_nodes = [9, 11, 15, 17]

    xy_face_nodes = [4, 22]
    xz_face_nodes = [12, 14]
    yz_face_nodes = [10, 16]

    body_nodes = [13]


def bary_coor1D(r):
    return r + 0.5

def bary_coor2D(s, t):
    m1, n1 = 1, 1
    m2, n2 = 1, 0
    m3, n3 = 0, 0
    m4, n4 = 0, 1

    N1 = 1./4*(1-s)*(1-t)
    N2 = 1./4*(1-s)*(1+t)
    N3 = 1./4*(1+s)*(1+t)
    N4 = 1./4*(1+s)*(1-t)
    ms = N1*m1 + N2*m2 + N3*m3 + N4*m4
    ns = N1*n1 + N2*n2 + N3*n3 + N4*n4
    return ms, ns

def bary_coor3D(s, t, q):
    x1, y1, z1 = 1, 1, 1
    x2, y2, z2 = 1, 1, 0
    x3, y3, z3 = 1, 0, 1
    x4, y4, z4 = 1, 0, 0
    x5, y5, z5 = 0, 1, 1
    x6, y6, z6 = 0, 1, 0
    x7, y7, z7 = 0, 0, 1
    x8, y8, z8 = 0, 0, 0

    N1 = 1./8*(1-s)*(1-t)*(1-q)
    N2 = 1./8*(1-s)*(1-t)*(1+q)
    N3 = 1./8*(1-s)*(1+t)*(1-q)
    N4 = 1./8*(1-s)*(1+t)*(1+q)
    N5 = 1./8*(1+s)*(1-t)*(1-q)
    N6 = 1./8*(1+s)*(1-t)*(1+q)
    N7 = 1./8*(1+s)*(1+t)*(1-q)
    N8 = 1./8*(1+s)*(1+t)*(1+q)
    
    xs = N1*x1 + N2*x2 + N3*x3 + N4*x4 + N5*x5 + N6*x6 + N7*x7 + N8*x8
    ys = N1*y1 + N2*y2 + N3*y3 + N4*y4 + N5*y5 + N6*y6 + N7*y7 + N8*y8
    zs = N1*z1 + N2*z2 + N3*z3 + N4*z4 + N5*z5 + N6*z6 + N7*z7 + N8*z8
    
    return xs, ys, zs

def ptb2nodes(new_ptb):

    nodes_n = nodesInit.copy()

    for n in x_edge_nodes:
        nodes_n[n,0] = bary_coor1D(new_ptb[n, 0]) 
    for n in y_edge_nodes:
        nodes_n[n,1] = bary_coor1D(new_ptb[n, 1]) 
    for n in z_edge_nodes:
        nodes_n[n,2] = bary_coor1D(new_ptb[n, 2])

    for n in xy_face_nodes:
        s, t = new_ptb[n,0], new_ptb[n,1]
        nodes_n[n,0], nodes_n[n,1] = bary_coor2D(s, t)

    for n in yz_face_nodes:
        s, t = new_ptb[n,1], new_ptb[n,2]
        nodes_n[n,1], nodes_n[n,2] = bary_coor2D(s, t)

    for n in xz_face_nodes:
        s, t = new_ptb[n,0], new_ptb[n,2]
        nodes_n[n,0], nodes_n[n,2] = bary_coor2D(s, t)

    for n in body_nodes:
        s, t, q= new_ptb[n,0], new_ptb[n,1], new_ptb[n,2]
        nodes_n[n,0], nodes_n[n,1], nodes_n[n,2] = bary_coor3D(s, t, q)
    return nodes_n


def adj_vec2array(adj_vec, a_row, a_col):

    num_sample = len(adj_vec)
    adj_arr = np.zeros([num_sample*numNodes, numNodes])

    # thresholding for post-processing
    adj_vec[adj_vec < 0.5] = 0.
    adj_vec[adj_vec > 0.5] = 1.

    for i in range(num_sample):
        recon_tmp = np.zeros([numNodes, numNodes])
        recon_tmp[a_row, a_col] = adj_vec[i,:].copy()
        adj_arr[i*numNodes:(i+1)*numNodes,:] = recon_tmp + recon_tmp.transpose()
    adj_arr[adj_arr > 0] = 1
    
    return adj_arr

def x_vec2array(x_vec, x_row, x_col):

    num_sample = len(x_vec)
    x_arr = np.zeros([num_sample*numNodes, dim])

    # thresholding for post-processing
    x_vec = np.clip(x_vec, -0.5, 0.5)
    x_vec[np.abs(x_vec) < 1e-3] = 0.

    for i in range(num_sample):
        
        x_recon_tmp = np.zeros([numNodes, dim])
        x_recon_tmp[x_row, x_col] = x_vec[i,:].copy()
        x_arr[i*numNodes:(i+1)*numNodes,:] = x_recon_tmp

    return x_arr


def check_bc_connected(adj, x, numUC):
    not_on_boundary = []
    for i in range(numUC):
        ex_adj = adj[i*numNodes:(i+1)*numNodes,:]
        ex_x = x[i*numNodes:(i+1)*numNodes,:]
        ex_adj += ex_adj.transpose()
        ex_adj [ex_adj > 0] = 1

        row = np.nonzero(ex_adj)[0]
        col = np.nonzero(ex_adj)[0]
        #   max_y = 1.
        bc_i = ex_x[row,1].max()
        bc_j = ex_x[col,1].max()
        if (bc_i < 1.) and (bc_j < 1.):
            not_on_boundary.append(i)

        #   min_y = 0.
        bc_i = np.min(ex_x[row,1])
        bc_j = np.min(ex_x[col,1])
        
        if (bc_i > 0.) and (bc_j > 0.):
            not_on_boundary.append(i) 
        #   min_z = 0.
        bc_i = np.min(ex_x[row,-1])
        bc_j = np.min(ex_x[col,-1])
        if (bc_i > 0.) and (bc_j > 0.):
            not_on_boundary.append(i)
        #   max_z = 1.
        bc_i = np.max(ex_x[row,-1])
        bc_j = np.max(ex_x[col,-1])
        if (bc_i < 1.) and (bc_j < 1.):
            not_on_boundary.append(i)
        #   max_x = 1.
        bc_i = np.max(ex_x[row,0])
        bc_j = np.max(ex_x[col,0])
        if (bc_i < 1.) and (bc_j < 1.):
            not_on_boundary.append(i)
        #   min_x = 0.
        bc_i = np.min(ex_x[row,0])
        bc_j = np.min(ex_x[col,0])
        if (bc_i > 0.) and (bc_j > 0.):
            not_on_boundary.append(i)
    return np.array(not_on_boundary)


def getObj(target_name, c_pred):
    C11, C12, C13 = c_pred[:,0], c_pred[:,1], c_pred[:,2]
    C22, C23, C33 = c_pred[:,3], c_pred[:,4], c_pred[:,5]
    C44, C55, C66 = c_pred[:,6], c_pred[:,7], c_pred[:,8]
    if target_name == 'v21':
        obj = (C12*C33 - C13*C23)/(C22*C33 - C23**2)
    elif target_name == 'v31':
        obj = (C12*C23 + C13*C22)/(C11*C33 - C12*C13)
    elif target_name == 'v32':
        obj = (C11*C23 - C12*C13)/(C11*C33 - C12*C13)
    else:
        print("Unknown target")
    return obj


def getKG(c_vec):
    K, G = np.zeros([len(c_vec)]),np.zeros([len(c_vec)])
    for i in range(len(c_vec)):
        c_k = c_vec[i,:]
        
        C11, C22, C33 = c_k[0], c_k[3], c_k[5]
        C12, C13, C23 = c_k[1], c_k[2], c_k[4]
        C44, C55, C66 = c_k[6], c_k[7], c_k[8] 
        K_iterk = 1/9.*((C11 + C22 + C33)+2*(C12 + C13 + C23))
        G_iterk = 1/15.*((C11 + C22 + C33)-(C12 + C13 + C23)+3*(C44+C55+C66))
        K[i] = K_iterk
        G[i] = G_iterk
    return K, G