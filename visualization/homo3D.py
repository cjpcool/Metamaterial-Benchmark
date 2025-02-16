# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:27:45 2024

@author: zj1283
"""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, LinearOperator, cg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spilu, spsolve

def homo3D(lx, ly, lz, lambda_, mu, voxel):
    """
    Calculate the effective stiff matrix of lattice structure from the voxel data.
    
    Parameters:
        lx, ly, lz (float): unit cell size.
        lambda_, mu (float): Material properties
        voxel (NxNxN array): structure of the lattice
    Returns:
        CH (6x6 array): the effective stiff matrix Cijkl.
    """
    # Initialize
    nelx, nely, nelz = voxel.shape
    dx = lx / nelx
    dy = ly / nely
    dz = lz / nelz
    nel = nelx * nely * nelz

    # Compute element stiffness matrices
    keLambda, keMu, feLambda, feMu = hexahedron(dx/2, dy/2, dz/2)

    # Node numbers and element degrees of freedom for full (not periodic) mesh
    nodenrs = np.arange(1, (1 + nelx) * (1 + nely) * (1 + nelz) + 1).reshape((1 + nelx, 1 + nely, 1 + nelz))
    edofVec = (3 * nodenrs[:-1, :-1, :-1] + 1).flatten()
    # addx = np.append([0, 1, 2], [3 * nelx + np.array([3, 4, 5, 0, 1, 2])])
    # addx = np.append(addx, [-3, -2, -1])
    addx = np.append([0, 1, 2], [3 * nelx + np.array([3, 4, 5, 0, 1, 2])])
    addx = np.append(addx, [-3, -2, -1])
    addxy = 3 * (nely + 1) * (nelx + 1) + addx
    edof = np.tile(edofVec[:, np.newaxis], (1, 24)) + np.tile(np.concatenate([addx, addxy]), (nel, 1))

    # Impose periodic boundary conditions
    nn = (nelx + 1) * (nely + 1) * (nelz + 1)  # Total number of nodes
    nnP = nelx * nely * nelz  # Total number of unique nodes
    nnPArray = np.arange(1, nnP + 1).reshape(nelx, nely, nelz)
    nnPArray = np.pad(nnPArray, ((0, 1), (0, 1), (0, 1)), mode='wrap')
    dofVector = np.zeros(3 * nn, dtype=int)
    dofVector[0::3] = 3 * nnPArray.flatten() - 2
    dofVector[1::3] = 3 * nnPArray.flatten() - 1
    dofVector[2::3] = 3 * nnPArray.flatten()
    #edof = dofVector[edof.flatten()].reshape(edof.shape)
    edof = edof-1
    edof = dofVector[edof]
    
    ndof = 3 * nnP

    # ASSEMBLE GLOBAL STIFFNESS MATRIX AND LOAD VECTORS
    # Indexing vectors
    iK = np.kron(edof, np.ones((24, 1))).T
    jK = np.kron(edof, np.ones((1, 24))).T
    # Material properties assigned to voxels with materials
    lambda_ = lambda_ * (voxel == 1)
    mu = mu * (voxel == 1)
    # The corresponding stiffness matrix entries
    sK = np.outer(keLambda.flatten('F'), lambda_.flatten('F')) + np.outer(keMu.flatten('F'), mu.flatten('F'))
    K = csr_matrix((sK.flatten('F'), (iK.flatten('F')-1, jK.flatten('F')-1)), shape=(ndof, ndof))
    K = 0.5 * (K + K.T)

    # Assembly three load cases corresponding to the three strain cases
    iF = np.tile(edof.T, (6, 1))
    jF = np.vstack([np.ones((24, nel)), 2 * np.ones((24, nel)), 3 * np.ones((24, nel)),
                    4 * np.ones((24, nel)), 5 * np.ones((24, nel)), 6 * np.ones((24, nel))])
    sF = np.outer(feLambda.flatten('F'), lambda_.flatten('F')) + np.outer(feMu.flatten('F'), mu.flatten('F'))
    F = csr_matrix((sF.flatten('F'), (iF.flatten('F')-1, jF.flatten('F')-1)), shape=(ndof, 6))

    # SOLUTION
    # solve by PCG method, remember to constrain one node
    activedofs = edof[voxel.flatten() == 1, :]
    activedofs = np.sort(np.unique(activedofs))
   # activedofs = activedofs-1
    X = np.zeros((ndof, 6))
    #L = splu(K[activedofs[3:], :][:, activedofs[3:]])
    
   # for i in range(6):
   #     X[activedofs[3:], i] = L.solve(F[activedofs[3:], i])
    K_act = K[activedofs[3:]-1, :][:, activedofs[3:]-1]
   #  free_dofs = activedofs[3:]
    # K_act = K[free_dofs, :][:, free_dofs]
    # K_act = K_act + sp.eye(K_act.shape[0]) * 1e-10
    K_act = csc_matrix(K_act)  # Convert matrix to CSC format
    M = LinearOperator(K_act.shape, spilu(K_act).solve)
    # Ensure b is a proper 1D vector
    for i in range(6):
        b = F[activedofs[3:]-1, i].toarray().flatten()  # Convert to 1D array if it's a sparse matrix
        X[activedofs[3:]-1, i], _ = cg(K_act, b, tol=1e-9, maxiter=300, M=M)


    # HOMOGENIZATION
    # The displacement vectors corresponding to the unit strain cases
    X0 = np.zeros((nel, 24, 6))
    # The element displacements for the six unit strains
    X0_e = np.zeros((24, 6))
    ke = keMu + keLambda  # Here the exact ratio does not matter, because
    fe = feMu + feLambda  # it is reflected in the load vector
    X0_e[np.array([3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]), :] = \
        np.linalg.solve(ke[np.array([3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]), 
                                  :][:, np.array([3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])], 
                         fe[np.array([3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]), :])
    
    for i in range(6):
        X0[:, :, i] = np.kron(X0_e[:, i].T[np.newaxis, :], np.ones((nel, 1)))

    CH = np.zeros((6, 6))
    volume = lx * ly * lz
    for i in range(6):
        for j in range(6):
            sum_L = np.dot((X0[:, :, i] - X.flatten('F')[edof.flatten('F')-1 + (i) * ndof].reshape(nnP, 24, order='F')), keLambda) * \
                     (X0[:, :, j] - X.flatten('F')[edof.flatten('F')-1 + (j) * ndof].reshape(nnP, 24, order='F'))
            sum_M = np.dot((X0[:, :, i] - X.flatten('F')[edof.flatten('F')-1 + (i) * ndof].reshape(nnP, 24, order='F')), keMu) * \
                      (X0[:, :, j] - X.flatten('F')[edof.flatten('F')-1 + (j)* ndof].reshape(nnP, 24, order='F'))
            sum_L = np.reshape(np.sum(sum_L, axis=1), (nelx, nely, nelz))
            sum_M = np.reshape(np.sum(sum_M, axis=1), (nelx, nely, nelz))
            CH[i, j] = np.sum(lambda_ * sum_L + mu * sum_M)
    CH = 1 / volume * CH
    return CH


def hexahedron(a, b, c):
    # Constitutive matrix contributions
    CMu = np.diag([2, 2, 2, 1, 1, 1])
    CLambda = np.zeros((6, 6))
    CLambda[0:3, 0:3] = 1
    
    # Three Gauss points in both directions
    xx = [-np.sqrt(3/5), 0, np.sqrt(3/5)]
    yy = xx
    zz = xx
    ww = [5/9, 8/9, 5/9]
    
    # Initialize
    keLambda = np.zeros((24, 24))
    keMu = np.zeros((24, 24))
    feLambda = np.zeros((24, 6))
    feMu = np.zeros((24, 6))
    
    for ii in range(len(xx)):
        for jj in range(len(yy)):
            for kk in range(len(zz)):
                # Integration point
                x = xx[ii]
                y = yy[jj]
                z = zz[kk]
                
                # Stress strain displacement matrix
                qx = np.array([-((y-1)*(z-1))/8, ((y-1)*(z-1))/8, -((y+1)*(z-1))/8,
                               ((y+1)*(z-1))/8, ((y-1)*(z+1))/8, -((y-1)*(z+1))/8,
                               ((y+1)*(z+1))/8, -((y+1)*(z+1))/8])
                
                qy = np.array([-((x-1)*(z-1))/8, ((x+1)*(z-1))/8, -((x+1)*(z-1))/8,
                               ((x-1)*(z-1))/8, ((x-1)*(z+1))/8, -((x+1)*(z+1))/8,
                               ((x+1)*(z+1))/8, -((x-1)*(z+1))/8])
                
                qz = np.array([-((x-1)*(y-1))/8, ((x+1)*(y-1))/8, -((x+1)*(y+1))/8,
                               ((x-1)*(y+1))/8, ((x-1)*(y-1))/8, -((x+1)*(y-1))/8,
                               ((x+1)*(y+1))/8, -((x-1)*(y+1))/8])
                
                # Jacobian
                J = np.array([qx, qy, qz]) @ np.array([[-a, a, a, -a, -a, a, a, -a],
                                                       [-b, -b, b, b, -b, -b, b, b],
                                                       [-c, -c, -c, -c, c, c, c, c]]).T
                qxyz = np.linalg.inv(J) @ np.array([qx, qy, qz])
                
                B_e = np.zeros((6, 3, 8))
                for i_B in range(8):
                    B_e[:, :, i_B] = np.array([
                        [qxyz[0, i_B], 0, 0],
                        [0, qxyz[1, i_B], 0],
                        [0, 0, qxyz[2, i_B]],
                        [qxyz[1, i_B], qxyz[0, i_B], 0],
                        [0, qxyz[2, i_B], qxyz[1, i_B]],
                        [qxyz[2, i_B], 0, qxyz[0, i_B]]
                    ])
                
                B = np.hstack([B_e[:, :, i] for i in range(8)])
                
                # Weight factor at this point
                weight = np.linalg.det(J) * ww[ii] * ww[jj] * ww[kk]
                
                # Element matrices
                keLambda += weight * B.T @ CLambda @ B
                keMu += weight * B.T @ CMu @ B
                
                # Element loads
                feLambda += weight * B.T @ CLambda
                feMu += weight * B.T @ CMu
    
    return keLambda, keMu, feLambda, feMu

# Example use
# voxel, Density = generate_voxel(30,'unit cell/octet.txt',0.2);
# CH = homo3D(1,1,1,0.5769,0.3846,voxel); 
# print(CH)