# %%
import warnings
import numpy as np
from typing import Union

try:
    import torch

    Tensor = torch.Tensor
except ImportError:
    Tensor = np.ndarray


# %% Enforce various symmetries on compliance tensors
def enforce_trigonal(S0: np.ndarray) -> np.ndarray:
    S = S0.copy()
    S[1, 1] = S[0, 0]
    S[1, 2] = S[0, 2] = S[2, 0] = S[2, 1]
    S[3, 3] = S[4, 4]
    S[5, 5] = 2 * (S[0, 0] - S[0, 1])
    S[1, 3] = -S[0, 3]
    S[3, 1] = S[1, 3]
    S[3, 0] = S[0, 3]
    S[4, 5] = 2 * S[0, 3]
    S[5, 4] = S[4, 5]
    S[1, 4] = -S[0, 4]
    S[4, 1] = S[1, 4]
    S[4, 0] = S[0, 4]
    S[3, 5] = 2 * S[1, 4]
    S[5, 3] = S[3, 5]
    return S


def enforce_tetragonal(S0: np.ndarray) -> np.ndarray:
    S = S0.copy()
    S = 0.5 * (S + S.T)
    S[1, 1] = S[0, 0]
    S[1, 2] = S[0, 2] = S[2, 0] = S[2, 1]
    S[3, 3] = S[4, 4]
    S[0:3, 3:6] = S[3:6, 0:3] = 0
    S[3, 4:6] = S[4:6, 3] = 0
    S[4, 5] = S[5, 4] = 0
    return S


def enforce_hexagonal(S0: np.ndarray) -> np.ndarray:
    S = S0.copy()
    S = 0.5 * (S + S.T)
    S[1, 1] = S[0, 0]
    S[1, 2] = S[0, 2] = S[2, 0] = S[2, 1]
    S[3, 3] = S[4, 4]
    S[5, 5] = 2 * (S[0, 0] - S[0, 1])
    S[0:3, 3:6] = S[3:6, 0:3] = 0
    S[3, 4:6] = S[4:6, 3] = 0
    S[4, 5] = S[5, 4] = 0
    return S


def enforce_orthotropic(S0: np.ndarray) -> np.ndarray:
    S = S0.copy()
    S = 0.5 * (S + S.T)
    S[0:3, 3:6] = S[3:6, 0:3] = 0
    S[3, 4:6] = S[4:6, 3] = 0
    S[4, 5] = S[5, 4] = 0
    return S


# %% Return compliance tensors for various symmetries
def orthotropic_S(Ex, Ey, Ez, Gyz, Gxz, Gxy, nuxy, nuyz, nuxz) -> np.ndarray:
    nuzy = nuyz * Ez / Ey
    nuzx = nuxz * Ez / Ex
    nuyx = nuxy * Ey / Ex
    S = np.array([[1 / Ex, -nuxy / Ex, -nuxz / Ex, 0, 0, 0],
                  [-nuyx / Ey, 1 / Ey, -nuyz / Ey, 0, 0, 0],
                  [-nuzx / Ez, -nuzy / Ez, 1 / Ez, 0, 0, 0],
                  [0, 0, 0, 1 / Gyz, 0, 0],
                  [0, 0, 0, 0, 1 / Gxz, 0],
                  [0, 0, 0, 0, 0, 1 / Gxy]])
    return S


def monoclinic_S(Ex, Ey, Ez, nuxy, nuxz, nuyz, nuxxyz, nuyyyz, nuzzyz, nuxzxy, Gyz, Gxz, Gxy) -> np.ndarray:
    s1 = 1 / Ex;
    s2 = 1 / Ey;
    s3 = 1 / Ez
    s4 = 1 / Gyz;
    s5 = 1 / Gxz;
    s6 = 1 / Gxy
    s7 = nuxy / Ex;
    s8 = nuxz / Ex;
    s9 = nuyz / Ey
    s10 = nuxxyz / Gyz;
    s11 = nuyyyz / Gxz;
    s12 = nuzzyz / Gxy;
    s13 = nuxzxy / Gxz
    S = np.array([[s1, -s7, -s8, s10, 0, 0],
                  [-s7, s2, -s9, s11, 0, 0],
                  [-s8, -s9, s3, s12, 0, 0],
                  [s10, s11, s12, s4, 0, 0],
                  [0, 0, 0, 0, s5, s13],
                  [0, 0, 0, 0, s13, s6]])
    return S


def tetragonal_S(Ex, Ez, nuxy, nuxz, Gyz, Gxy) -> np.ndarray:
    s1 = 1 / Ex
    s2 = 1 / Ez
    s3 = 1 / Gyz
    s4 = 1 / Gxy
    s5 = nuxy / Ex
    s6 = nuxz / Ex
    S = np.array([[s1, -s5, -s6, 0, 0, 0],
                  [-s5, s1, -s6, 0, 0, 0],
                  [-s6, -s6, s2, 0, 0, 0],
                  [0, 0, 0, s3, 0, 0],
                  [0, 0, 0, 0, s3, 0],
                  [0, 0, 0, 0, 0, s4]])
    return S


def trigonal_S(Ex, Ez, nuxy, nuxz, Gyz, nuxxyz) -> np.ndarray:
    s1 = 1 / Ex
    s2 = 1 / Ez
    s3 = 1 / Gyz
    s5 = nuxy / Ex
    s6 = nuxz / Ex
    s4 = nuxxyz / Gyz
    S = np.array([[s1, -s5, -s6, s4, 0, 0],
                  [-s5, s1, -s6, -s4, 0, 0],
                  [-s6, -s6, s2, 0, 0, 0],
                  [s4, -s4, 0, s3, 0, 0],
                  [0, 0, 0, 0, s3, 2 * s4],
                  [0, 0, 0, 0, 2 * s4, 2 * (s1 + s5)]])
    return S


def cubic_S(E, nu, G) -> np.ndarray:
    s1 = 1 / E
    s2 = nu / E
    s3 = 1 / G
    S = np.array([[s1, -s2, -s2, 0, 0, 0],
                  [-s2, s1, -s2, 0, 0, 0],
                  [-s2, -s2, s1, 0, 0, 0],
                  [0, 0, 0, s3, 0, 0],
                  [0, 0, 0, 0, s3, 0],
                  [0, 0, 0, 0, 0, s3]])
    return S


def hexagonal_S(Ex, Ez, nuxy, nuxz, Gyz) -> np.ndarray:
    s1 = 1 / Ex
    s2 = nuxy / Ex
    s3 = nuxz / Ex
    s4 = 1 / Ez
    s5 = 1 / Gyz
    S = np.array([[s1, -s2, -s3, 0, 0, 0],
                  [-s2, s1, -s3, 0, 0, 0],
                  [-s3, -s3, s4, 0, 0, 0],
                  [0, 0, 0, s5, 0, 0],
                  [0, 0, 0, 0, s5, 0],
                  [0, 0, 0, 0, 0, 2 * (s1 + s2)]])
    return S


def isotropic_S(E=1, nu=0.3) -> np.ndarray:
    G = 0.5 * E / (1 + nu)
    S = np.array([[1 / E, -nu / E, -nu / E, 0, 0, 0],
                  [-nu / E, 1 / E, -nu / E, 0, 0, 0],
                  [-nu / E, -nu / E, 1 / E, 0, 0, 0],
                  [0, 0, 0, 1 / G, 0, 0],
                  [0, 0, 0, 0, 1 / G, 0],
                  [0, 0, 0, 0, 0, 1 / G]])
    return S


# %% Rotation functions
def rotate_4th_order(T: np.ndarray, g: np.ndarray):
    out = np.einsum('ia, jb, kc, ld, abcd->ijkl', g, g, g, g, T)
    return out


def rotate_Voigt_stiffness(C: np.ndarray, R: np.ndarray):
    K1 = R ** 2
    K2 = R[:, [1, 2, 0]] * R[:, [2, 0, 1]]
    K3 = R[[1, 2, 0], :] * R[[2, 0, 1], :]
    K4 = R[[1, 2, 0], :][:, [1, 2, 0]] * R[[2, 0, 1], :][:, [2, 0, 1]] + R[[1, 2, 0], :][:, [2, 0, 1]] * R[[2, 0, 1],
                                                                                                         :][:,
                                                                                                         [1, 2, 0]]
    K = np.block([
        [K1, 2 * K2],
        [K3, K4]
    ])
    return K @ C @ (K.T)


def rotate_Voigt_compliance(S: np.ndarray, R: np.ndarray):
    K1 = R ** 2
    K2 = R[:, [1, 2, 0]] * R[:, [2, 0, 1]]
    K3 = R[[1, 2, 0], :] * R[[2, 0, 1], :]
    K4 = R[[1, 2, 0], :][:, [1, 2, 0]] * R[[2, 0, 1], :][:, [2, 0, 1]] + R[[1, 2, 0], :][:, [2, 0, 1]] * R[[2, 0, 1],
                                                                                                         :][:,
                                                                                                         [1, 2, 0]]
    K = np.block([
        [K1, 2 * K2],
        [K3, K4]
    ])
    K_T = np.block([
        [K1, K2],
        [2 * K3, K4]
    ])
    return K_T @ S @ np.linalg.inv(K)


# %%
def compliance_Voigt_to_4th_order(S: np.ndarray):
    if S.ndim == 2:
        _S = S.reshape((1, 6, 6))
    elif S.ndim == 3:
        _S = S
    # Convert S_ij to S_abcd
    assert np.allclose(_S, _S.transpose((0, 2, 1))), "S is not symmetric"
    S_4 = np.zeros((_S.shape[0], 3, 3, 3, 3))
    for i in range(6):
        if i < 3:
            a = b = i
        elif i == 3:
            a = 1;
            b = 2
        elif i == 4:
            a = 0;
            b = 2
        else:
            a = 0;
            b = 1
        for j in range(i, 6):
            if j < 3:
                c = d = j
            elif j == 3:
                c = 1;
                d = 2
            elif j == 4:
                c = 0;
                d = 2
            else:
                c = 0;
                d = 1
            Sij = _S[:, i, j]
            Sij = Sij if i < 3 else Sij / 2
            Sij = Sij if j < 3 else Sij / 2
            S_4[:, a, b, c, d] = Sij
            S_4[:, a, b, d, c] = Sij
            S_4[:, b, a, c, d] = Sij
            S_4[:, b, a, d, c] = Sij
            S_4[:, c, d, a, b] = Sij
            S_4[:, c, d, b, a] = Sij
            S_4[:, d, c, a, b] = Sij
            S_4[:, d, c, b, a] = Sij
    if S.ndim == 2:
        return S_4[0]
    else:
        return S_4


# %%
def compliance_4th_order_to_Voigt(S: np.ndarray):
    if S.ndim == 4:
        _S = S.reshape((1, 3, 3, 3, 3))
    elif S.ndim == 5:
        _S = S
    # Convert S_abcd to S_ij
    assert np.allclose(_S, np.transpose(_S, (0, 1, 2, 4, 3)))
    assert np.allclose(_S, np.transpose(_S, (0, 2, 1, 3, 4)))
    assert np.allclose(_S, np.transpose(_S, (0, 3, 4, 1, 2)))
    S_2 = np.zeros((_S.shape[0], 6, 6))
    for i in range(6):
        if i < 3:
            a = b = i
        elif i == 3:
            a = 1;
            b = 2
        elif i == 4:
            a = 0;
            b = 2
        else:
            a = 0;
            b = 1
        for j in range(i, 6):
            if j < 3:
                c = d = j
            elif j == 3:
                c = 1;
                d = 2
            elif j == 4:
                c = 0;
                d = 2
            else:
                c = 0;
                d = 1
            Sij = _S[:, a, b, c, d] if j < 3 else 2 * _S[:, a, b, c, d]
            Sij = Sij if i < 3 else 2 * Sij
            S_2[:, i, j] = Sij
            S_2[:, j, i] = Sij
    if S.ndim == 4:
        return S_2[0]
    else:
        return S_2


# %%
def stiffness_4th_order_to_Voigt(C: np.ndarray) -> np.ndarray:
    if C.ndim == 5:
        _C = C
    elif C.ndim == 4:
        _C = C.reshape((1, 3, 3, 3, 3))
    else:
        raise ValueError(f'Wrong shape of C: C.shape={C.shape}')
    C_2 = np.zeros((_C.shape[0], 6, 6))
    for i in range(6):
        if i < 3:
            a = b = i
        elif i == 3:
            a = 1;
            b = 2
        elif i == 4:
            a = 0;
            b = 2
        else:
            a = 0;
            b = 1
        for j in range(i, 6):
            if j < 3:
                c = d = j
            elif j == 3:
                c = 1;
                d = 2
            elif j == 4:
                c = 0;
                d = 2
            else:
                c = 0;
                d = 1
            Cij = _C[:, a, b, c, d]
            C_2[:, i, j] = Cij
            C_2[:, j, i] = Cij
    if C.ndim == 5:
        return C_2
    else:
        return C_2[0, :, :]


# %%
def stiffness_Voigt_to_4th_order(C: np.ndarray):
    # convert C_ij to C_abcd
    if C.ndim == 3:
        _C = C
    elif C.ndim == 2:
        _C = C.reshape((1, 6, 6))
    assert np.allclose(_C, _C.transpose((0, 2, 1)))
    C4 = np.zeros((_C.shape[0], 3, 3, 3, 3))
    # 1-based continuum mechanics notation
    for i in range(1, 7):
        if i <= 3:
            a = b = i
        elif i == 4:
            a = 2;
            b = 3
        elif i == 5:
            a = 1;
            b = 3
        elif i == 6:
            a = 1;
            b = 2

        for j in range(i, 7):
            if j <= 3:
                c = d = j
            elif j == 4:
                c = 2;
                d = 3
            elif j == 5:
                c = 1;
                d = 3
            elif j == 6:
                c = 1;
                d = 2

            C4[:, a - 1, b - 1, c - 1, d - 1] = _C[:, i - 1, j - 1]
            C4[:, b - 1, a - 1, c - 1, d - 1] = _C[:, i - 1, j - 1]
            C4[:, a - 1, b - 1, d - 1, c - 1] = _C[:, i - 1, j - 1]
            C4[:, b - 1, a - 1, d - 1, c - 1] = _C[:, i - 1, j - 1]
            C4[:, c - 1, d - 1, a - 1, b - 1] = _C[:, i - 1, j - 1]
            C4[:, c - 1, d - 1, b - 1, a - 1] = _C[:, i - 1, j - 1]
            C4[:, d - 1, c - 1, a - 1, b - 1] = _C[:, i - 1, j - 1]
            C4[:, d - 1, c - 1, b - 1, a - 1] = _C[:, i - 1, j - 1]
    if C.ndim == 3:
        return C4
    else:
        return C4[0, :, :, :, :]


def stiffness_Voigt_to_Mandel(C: np.ndarray) -> np.ndarray:
    s2 = np.sqrt(2)
    mask = np.block([[np.ones((3, 3)), s2 * np.ones((3, 3))],
                     [s2 * np.ones((3, 3)), 2 * np.ones((3, 3))]])
    C_2 = C[..., :, :] * mask
    return C_2


def stiffness_Mandel_to_Voigt(C: np.ndarray) -> np.ndarray:
    s2 = np.sqrt(2)
    mask = np.block([[np.ones((3, 3)), s2 * np.ones((3, 3))],
                     [s2 * np.ones((3, 3)), 2 * np.ones((3, 3))]])
    C_2 = C[..., :, :] / mask
    return C_2


def compliance_Voigt_to_Mandel(S: np.ndarray) -> np.ndarray:
    s2 = np.sqrt(2)
    mask = np.block([[np.ones((3, 3)), s2 * np.ones((3, 3))],
                     [s2 * np.ones((3, 3)), 2 * np.ones((3, 3))]])
    S_2 = S[..., :, :] / mask
    return S_2


def compliance_Mandel_to_Voigt(S: np.ndarray) -> np.ndarray:
    s2 = np.sqrt(2)
    mask = np.block([[np.ones((3, 3)), s2 * np.ones((3, 3))],
                     [s2 * np.ones((3, 3)), 2 * np.ones((3, 3))]])
    S_2 = S[..., :, :] * mask
    return S_2


def numpy_cart_4_to_Mandel(_C: np.ndarray) -> np.ndarray:
    s2 = np.sqrt(2)
    mask = np.block([[np.ones((3, 3)), s2 * np.ones((3, 3))],
                     [s2 * np.ones((3, 3)), 2 * np.ones((3, 3))]])
    if _C.ndim == 4:
        C = _C[None, ...]  # add batch dimension
    else:
        C = _C
    C_2 = np.zeros((C.shape[0], 6, 6))
    _cart_4_tensor_to_Mandel_inplace(C, C_2, mask)
    if _C.ndim == 4:
        C_2 = C_2[0, ...]  # remove batch dimension
    return C_2


def numpy_Mandel_to_cart_4(C: np.ndarray) -> np.ndarray:
    # convert C_ij to C_abcd
    if C.ndim == 3:
        _C = C
    elif C.ndim == 2:
        _C = C[None, ...]  # add batch dimension
    C4 = np.zeros((_C.shape[0], 3, 3, 3, 3))
    _Mandel_to_cart_4_inplace(_C, C4)

    if C.ndim == 2:
        C4 = C4[0, ...]  # remove batch dimension
    return C4


# %% Young's modulus in a specific direction
def Youngs_modulus(S: np.ndarray, d: np.ndarray):
    assert S.shape == (3, 3, 3, 3)
    if d.ndim == 1:
        assert np.allclose(np.linalg.norm(d), 1.0)
        Sc = np.einsum('i,j,k,l,ijkl', d, d, d, d, S)
        return 1 / Sc
    elif d.ndim == 2:
        assert d.shape[1] == 3
        assert np.allclose(np.linalg.norm(d, axis=1), np.ones(d.shape[0]))
        Sc = np.einsum('ai,aj,ak,al,ijkl->a', d, d, d, d, S)
        return 1 / Sc


# %% Scaling exponents
def scaling_law_fit(youngs_moduli: dict):
    x = list(youngs_moduli.keys())
    y = np.row_stack([youngs_moduli[rel_dens] for rel_dens in x])

    x = np.log10(x)
    y = np.log10(y)
    fit = np.polyfit(x, y, deg=1)
    exponents = fit[0, :]
    constants = 10 ** (fit[1, :])

    return exponents, constants


# %%  FUNCTIONS SHARED BETWEEN NUMPY AND TORCH ###
def _cart_4_tensor_to_Mandel_inplace(C4: Union[np.ndarray, Tensor], C_2: Union[np.ndarray, Tensor],
                                     mask: Union[np.ndarray, Tensor]):
    for i in range(6):
        if i < 3:
            a = b = i
        elif i == 3:
            a = 1;
            b = 2
        elif i == 4:
            a = 0;
            b = 2
        else:
            a = 0;
            b = 1
        for j in range(i, 6):
            if j < 3:
                c = d = j
            elif j == 3:
                c = 1;
                d = 2
            elif j == 4:
                c = 0;
                d = 2
            else:
                c = 0;
                d = 1
            Cij = C4[..., a, b, c, d]
            C_2[..., i, j] = Cij
            C_2[..., j, i] = Cij
    C_2[..., :, :] *= mask


def _Mandel_to_cart_4_inplace(C: Union[np.ndarray, Tensor], C4: Union[np.ndarray, Tensor]):
    for i in range(1, 7):
        if i <= 3:
            a = b = i
        elif i == 4:
            a = 2;
            b = 3
        elif i == 5:
            a = 1;
            b = 3
        elif i == 6:
            a = 1;
            b = 2

        for j in range(i, 7):
            if j <= 3:
                c = d = j
            elif j == 4:
                c = 2;
                d = 3
            elif j == 5:
                c = 1;
                d = 3
            elif j == 6:
                c = 1;
                d = 2

            val = C[..., i - 1, j - 1]
            if i > 3:
                val = val / np.sqrt(2)
            if j > 3:
                val = val / np.sqrt(2)

            C4[..., a - 1, b - 1, c - 1, d - 1] = val
            C4[..., b - 1, a - 1, c - 1, d - 1] = val
            C4[..., a - 1, b - 1, d - 1, c - 1] = val
            C4[..., c - 1, d - 1, a - 1, b - 1] = val
            C4[..., b - 1, a - 1, d - 1, c - 1] = val
            C4[..., c - 1, d - 1, b - 1, a - 1] = val
            C4[..., d - 1, c - 1, a - 1, b - 1] = val
            C4[..., d - 1, c - 1, b - 1, a - 1] = val


# %%  TORCH FUNCTIONS ###
#########################
try:
    import torch


    def stiffness_cart_4_to_Mandel(_C: torch.Tensor) -> torch.Tensor:
        return tensor_cart_4_to_Mandel(_C)


    def compliance_cart_4_to_Mandel(_S: torch.Tensor) -> torch.Tensor:
        return tensor_cart_4_to_Mandel(_S)


    def stiffness_Mandel_to_cart_4(_C: torch.Tensor) -> torch.Tensor:
        return Mandel_to_cart_4_tensor(_C)


    def compliance_Mandel_to_cart_4(_S: torch.Tensor) -> torch.Tensor:
        return Mandel_to_cart_4_tensor(_S)


    def tensor_cart_4_to_Mandel(_C: torch.Tensor) -> torch.Tensor:
        s2 = np.sqrt(2)
        mask = torch.tensor([[1, 1, 1, s2, s2, s2],
                             [1, 1, 1, s2, s2, s2],
                             [1, 1, 1, s2, s2, s2],
                             [s2, s2, s2, 2, 2, 2],
                             [s2, s2, s2, 2, 2, 2],
                             [s2, s2, s2, 2, 2, 2]], device=_C.device, dtype=_C.dtype)
        if _C.dim() == 4:
            C = _C.unsqueeze(0)
        else:
            C = _C
        C_2 = _C.new_zeros((C.size(0), 6, 6))
        _cart_4_tensor_to_Mandel_inplace(C, C_2, mask)
        if _C.dim() == 4:
            C_2.squeeze_(0)
        return C_2


    def Mandel_to_cart_4_tensor(C: torch.Tensor) -> torch.Tensor:
        # convert C_ij to C_abcd
        if C.ndim == 3:
            _C = C
        elif C.ndim == 2:
            _C = C.unsqueeze(0)
        C4 = torch.zeros((_C.shape[0], 3, 3, 3, 3), device=C.device, dtype=C.dtype)
        _Mandel_to_cart_4_inplace(_C, C4)

        if C.ndim == 2:
            C4.squeeze_(0)
        return C4

except ImportError:
    warnings.warn('torch not imported. Some functions will not be available.')