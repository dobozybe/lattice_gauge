import numpy as np
import random
import warnings
from numba import *
set_num_threads(4)

warnings.filterwarnings(
    "ignore",
    message="Casting complex values to real discards the imaginary part"
)

@njit
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

@njit
def jit_2x2_mult(A,B):
    output = np.zeros((2,2), dtype = np.complex128)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                output[i,j] += A[i,k] * B[k,j]
    return output

@njit(parallel=True)
def jit_2x2_mult_batch(A,B): #ashape, bshape = (n,k,2,2)
    inshape = A.shape
    outarray = np.empty(inshape, dtype=np.complex128)
    for nodeindex in prange(inshape[0]):
        for direction in range(inshape[1]):
            outarray[nodeindex, direction] =jit_2x2_mult(A[nodeindex, direction], B[nodeindex, direction])
    return outarray


@njit(parallel=True, fastmath=True)
def jit_2x2_mult_batch_fast(A, B):
    """Optimized batch 2x2 matrix multiplication"""
    n, k = A.shape[0], A.shape[1]
    outarray = np.empty((n, k, 2, 2), dtype=np.complex128)

    for nodeindex in prange(n):
        for direction in range(k):
            # Unroll the entire multiplication inline
            outarray[nodeindex, direction, 0, 0] = (
                    A[nodeindex, direction, 0, 0] * B[nodeindex, direction, 0, 0] +
                    A[nodeindex, direction, 0, 1] * B[nodeindex, direction, 1, 0]
            )
            outarray[nodeindex, direction, 0, 1] = (
                    A[nodeindex, direction, 0, 0] * B[nodeindex, direction, 0, 1] +
                    A[nodeindex, direction, 0, 1] * B[nodeindex, direction, 1, 1]
            )
            outarray[nodeindex, direction, 1, 0] = (
                    A[nodeindex, direction, 1, 0] * B[nodeindex, direction, 0, 0] +
                    A[nodeindex, direction, 1, 1] * B[nodeindex, direction, 1, 0]
            )
            outarray[nodeindex, direction, 1, 1] = (
                    A[nodeindex, direction, 1, 0] * B[nodeindex, direction, 0, 1] +
                    A[nodeindex, direction, 1, 1] * B[nodeindex, direction, 1, 1]
            )

    return outarray
@njit
def jit_2x2_add(A,B):
    output = np.zeros((2, 2), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            output[i, j] = A[i, j] + B[i, j]
    return output
@njit
def jit_2x2_scale(A, a):
    output = np.zeros((2, 2), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            output[i,j] = a * A[i,j]
    return output


@njit
def jit_2x2_trace(A):
    return A[0,0]+A[1,1]

@njit
def jit_2x2_dagger(A):
    output = A.copy()
    for i in range(2):
        for j in range(2):
            output[i,j] = np.conj(A[j,i])
    return output

@njit(parallel=True)
def jit_su2_exp(inputarray): #input shape: (N, d, 2,2)
    inshape = inputarray.shape
    gen0 = jit_2x2_dagger(lie_gens[0])
    gen1 = jit_2x2_dagger(lie_gens[1])
    gen2 = jit_2x2_dagger(lie_gens[2])
    outarray = np.ascontiguousarray(np.zeros(inshape, dtype=np.complex128))

    for nodeindex in prange(inshape[0]):
        paramarray = np.ascontiguousarray(np.zeros((3,), dtype=np.complex128))
        for direction in range(inshape[1]):
            sin_term = np.zeros((2,2), dtype=np.complex128)
            thismatrix = inputarray[nodeindex, direction]
            param1 = jit_2x2_trace(jit_2x2_mult(thismatrix, gen0))/2
            param2 = jit_2x2_trace(jit_2x2_mult(thismatrix, gen1))/2
            param3 = jit_2x2_trace(jit_2x2_mult(thismatrix, gen2))/2
            paramarray[0]=param1
            paramarray[1]=param2
            paramarray[2]=param3
            normparams = np.real(np.sqrt(param1**2+param2**2+param3**2))
            modded_normparams = np.mod(normparams, 2 * np.pi)
            if normparams == 0:
                normparams = 1e-12
            for i in range(3):
                paramarray[i] = paramarray[i]/normparams
            cos_term = jit_2x2_scale(get_identity(), np.cos(modded_normparams))
            for j in range(3):
                sin_term =jit_2x2_add(sin_term, jit_2x2_scale(lie_gens[j], paramarray[j]))
            sin_term = jit_2x2_scale(sin_term, np.sin(modded_normparams))
            final = jit_2x2_add(cos_term,sin_term)
            for k in range(2):
                for l in range(2):
                    outarray[nodeindex, direction,k,l] = final[k,l]
    return outarray




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


def extra_action_term(action):
    a = 0.1
    S = 39.48
    scale = 1
    prefactor = 1/(np.sqrt(np.pi) * a)
    return 0
    return scale * prefactor * (np.exp(-(action - S + 0.1*S)**2/a**2) + np.exp(-(action - S - 0.1*S)**2/a**2))


def extra_action_term_derivative(action):
    a = 0.1
    S = 39.48
    scale = 1
    prefactor = 1 / (np.sqrt(np.pi) * a)
    return 0
    return -2 * scale * prefactor * (
        (action - S + 0.1 * S)* np.exp(-(action - S + 0.1 * S) ** 2 / a ** 2) + (action - S - 0.1 * S)*np.exp(-(action - S - 0.1 * S) ** 2 / a ** 2)
    )





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


def project(matrixarray): #projects a matrix into su(2) lie algebra
    antisym = (matrixarray - np.transpose(matrixarray.conj(), axes=(*range(matrixarray.ndim - 2), -1, -2)))
    return (1 / 2) * antisym - (1 / 4) * np.trace(antisym, axis1 = -2, axis2 = -1)[...,np.newaxis, np.newaxis] * np.array([[1, 0], [0, 1]])

def make_plaquette_array(config, plaquette_index_array):
    link_array = config[0]
    links_shape = np.shape(link_array)
    num_nodes = links_shape[0]
    dim = links_shape[1]

    plaquette_matrix_array = np.zeros((num_nodes, dim, dim,4, 2,2), dtype = np.complex128)
    mask = np.ones(plaquette_index_array.shape[:3], dtype=np.bool_)
    diag_indices = np.arange(min(plaquette_index_array.shape[1], plaquette_index_array.shape[2]))
    mask[:, diag_indices, diag_indices] = False

    #plaquette index array[mask] shape is (N * 4 * 3, 4,2). Transpose to (2,4,N*4*3)

    result = link_array[tuple(plaquette_index_array[mask].T)] #should give (4, N*4*3) array of 2x2 matricies (links for each spot), so (4, N*4*3,2,2)



    result_reordered = result.transpose(1,0,2,3)


    plaquette_matrix_array[mask] = result_reordered

    return plaquette_matrix_array


