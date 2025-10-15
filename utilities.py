import numpy as np
import random
import warnings


warnings.filterwarnings(
    "ignore",
    message="Casting complex values to real discards the imaginary part"
)

def get_identity():
    return np.diag(np.full(2, 1))


lie_gens = np.stack([ #i sigma_n where sigma_i are the paulis.
    np.array([[0, 1], [-1, 0]]),
    np.array([[0, 1j], [1j, 0]]),
    np.array([[1j, 0], [0, -1j]])
])


def su2_exp(matrixarray):
    array_shape = np.shape(matrixarray)
    matrixarray = matrixarray.reshape(-1,2,2)
    gen0 = np.broadcast_to(lie_gens[0].conj().T, matrixarray.shape)
    gen1 = np.broadcast_to(lie_gens[1].conj().T, matrixarray.shape)
    gen2 = np.broadcast_to(lie_gens[2].conj().T, matrixarray.shape)
    param1 = np.trace(matrixarray @ gen0, axis1=1, axis2=2)/2
    param2 = np.trace(matrixarray @ gen1, axis1=1, axis2=2)/2
    param3 = np.trace(matrixarray @ gen2, axis1=1, axis2=2)/2

    paramarray = np.stack([param1, param2, param3]).T
    normparams = np.sqrt(param1 ** 2 + param2 ** 2 + param3 ** 2)
    """    if np.max(np.imag(normparams))*(g**2) > 0.01:
        print("error! Normparams in su2_exp is imaginary!", np.max(np.imag(normparams)))
        return None"""
    normparams = np.float64(normparams)
    modded_normparams = np.mod(normparams, 2 * np.pi)
    normparams_safe = np.where(normparams == 0, 1e-12, normparams)
    normed_paramarray = paramarray/normparams_safe[:, np.newaxis]
    exp = np.eye(2)[np.newaxis, :, :] * np.cos(modded_normparams)[:, np.newaxis, np.newaxis]+np.einsum(
        "ij, jkl->ikl", normed_paramarray, lie_gens
    ) * np.sin(modded_normparams)[:, np.newaxis, np.newaxis]
    exp = exp.reshape(array_shape)
    return exp


def random_su2_matrix():
    # Random real numbers for the basis coefficients
    a1, a2, a3 = np.random.randn(3)

    # su(2) basis using Pauli matrices (times i)
    sigma1 = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma3 = np.array([[1, 0], [0, -1]], dtype=complex)

    # Linear combination: X = i(a1 * σ1 + a2 * σ2 + a3 * σ3)
    su2_matrix = 1j * (a1 * sigma1 + a2 * sigma2 + a3 * sigma3)
    return su2_matrix


x = np.array([random_su2_matrix()])




def randomSU2():
    alpha = random.random() * 2 * np.pi
    beta = random.random() * 2 * np.pi
    gamma = random.random() * 2 * np.pi
    a = np.exp(alpha * 1j) * np.sin(beta)
    b = np.exp(gamma * 1j) * np.cos(beta)
    matrix = np.array([[a, -np.conj(b)], [b, np.conj(a)]])
    return matrix


def inSU2(matrix, tol=1e-10):
    det_close = np.allclose(np.linalg.det(matrix), 1.0, atol=tol)
    unitary = np.allclose(matrix @ matrix.conj().T, np.eye(2), atol=tol)
    return det_close and unitary


def project(matrixarray):
    antisym = (matrixarray - np.transpose(matrixarray.conj(), axes=(*range(matrixarray.ndim - 2), -1, -2)))
    return (1 / 2) * antisym - (1 / 4) * np.trace(antisym, axis1 = -2, axis2 = -1)[...,np.newaxis, np.newaxis] * np.array([[1, 0], [0, 1]])

def make_plaquette_array(config, plaquette_index_array):
    link_array = config[0]
    links_shape = np.shape(link_array)
    num_nodes = links_shape[0]
    dim = links_shape[1]

    plaquette_matrix_array = np.zeros((num_nodes, dim, dim,4, 2,2), dtype = np.complex128)
    mask = np.ones(plaquette_index_array.shape[:3], dtype=bool)
    diag_indices = np.arange(min(plaquette_index_array.shape[1], plaquette_index_array.shape[2]))
    mask[:, diag_indices, diag_indices] = False

    #plaquette index array[mask] shape is (N * 4 * 3, 4,2). Transpose to (2,4,N*4*3)

    result = link_array[tuple(plaquette_index_array[mask].T)] #should give (4, N*4*3) array of 2x2 matricies (links for each spot), so (4, N*4*3,2,2)



    result_reordered = result.transpose(1,0,2,3)


    plaquette_matrix_array[mask] = result_reordered

    return plaquette_matrix_array


