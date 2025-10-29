
from numba import cuda, complex128, float64
import time
import math

@cuda.jit(device=True, inline=True)
def get_identity(out):
    out[0,0] = 1.0 + 0.0j
    out[0,1] = 0.0 + 0.0j
    out[1,0] = 0.0 + 0.0j
    out[1,1] = 1.0 + 0.0j

@cuda.jit(device=True)
def matmul_2x2_cuda(A, B, out):
    """
    Multiply two 2x2 matrices: out = A @ B
    Safe for any of A, B, and out being the same array.
    """
    # Store original values from both A and B
    a00 = A[0,0]
    a01 = A[0,1]
    a10 = A[1,0]
    a11 = A[1,1]

    b00 = B[0,0]
    b01 = B[0,1]
    b10 = B[1,0]
    b11 = B[1,1]

    # Compute output using cached values
    out[0,0] = a00*b00 + a01*b10
    out[0,1] = a00*b01 + a01*b11
    out[1,0] = a10*b00 + a11*b10
    out[1,1] = a10*b01 + a11*b11

@cuda.jit(device=True)
def add_2x2_cuda(A, B, out):
    """Add two 2x2 matrices: out = A + B"""
    out[0, 0] = A[0, 0] + B[0, 0]
    out[0, 1] = A[0, 1] + B[0, 1]
    out[1, 0] = A[1, 0] + B[1, 0]
    out[1, 1] = A[1, 1] + B[1, 1]

@cuda.jit(device=True)
def scale_2x2_cuda(A, scalar, out):
    """Scale a 2x2 matrix: out = scalar * A"""
    out[0, 0] = scalar * A[0, 0]
    out[0, 1] = scalar * A[0, 1]
    out[1, 0] = scalar * A[1, 0]
    out[1, 1] = scalar * A[1, 1]

@cuda.jit(device=True)
def dagger_2x2_cuda(A, out):
    A0 = A[0, 0]
    A1 = A[0, 1]
    A2 = A[1, 0]
    A3 = A[1, 1]


    out[0, 0] = complex(A0.real, -A0.imag)
    out[0, 1] = complex(A2.real, -A2.imag)
    out[1, 0] = complex(A1.real, -A1.imag)
    out[1, 1] = complex(A3.real, -A3.imag)


@cuda.jit(device=True)
def trace_2x2_cuda(A):
    """Returns trace of 2x2 matrix"""
    return A[0, 0] + A[1, 1]

@cuda.jit(device=True, inline=True)
def su2_exp_cuda(inputmatrix, outputmatrix, lie_gens, idx): #input shape: (2,2)

    temp = cuda.local.array((2, 2), dtype=complex128)


    dagger_2x2_cuda(lie_gens[0], temp)
    matmul_2x2_cuda(inputmatrix, temp, temp)
    param0 = trace_2x2_cuda(temp)/2

    dagger_2x2_cuda(lie_gens[1], temp)
    matmul_2x2_cuda(inputmatrix, temp, temp)
    param1 = trace_2x2_cuda(temp) / 2

    dagger_2x2_cuda(lie_gens[2], temp)
    matmul_2x2_cuda(inputmatrix, temp, temp)
    param2 = trace_2x2_cuda(temp) / 2


    normparams = math.sqrt(
        abs(param0) ** 2 + abs(param1) ** 2 + abs(param2) ** 2
    )
    if normparams == 0:
        normparams = 1e-12

    modded_normparams = math.fmod(normparams, 2 * math.pi)

    param0/=normparams
    param1/=normparams
    param2/=normparams




    cos_val = math.cos(modded_normparams)
    sin_val = math.sin(modded_normparams)

    for i in range(2):
        for j in range(2):
            temp[i,j]=param0 * lie_gens[0,i,j] + param1 * lie_gens[1,i,j] + param2 * lie_gens[2,i,j]




    outputmatrix[0,0] = cos_val + sin_val * temp[0,0]
    outputmatrix[0, 1] = sin_val * temp[0, 1]
    outputmatrix[1, 0] = sin_val * temp[1, 0]
    outputmatrix[1, 1] = cos_val + sin_val * temp[1,1]

    """temp2 = cuda.local.array((2, 2), dtype=complex128)

    dagger_2x2_cuda(lie_gens[0], temp2)
    matmul_2x2_cuda(outputmatrix, temp2, temp2)
    param0 = trace_2x2_cuda(temp2) / 2

    dagger_2x2_cuda(lie_gens[1], temp2)
    matmul_2x2_cuda(outputmatrix, temp2, temp2)
    param1 = trace_2x2_cuda(temp2) / 2

    dagger_2x2_cuda(lie_gens[2], temp2)
    matmul_2x2_cuda(outputmatrix, temp2, temp2)
    param2 = trace_2x2_cuda(temp2) / 2

    if idx == 0:
        print("Before output check")
        print("param0 real:", param0.real)
        print("param1 real:", param1.real)
        print("param2 real:", param2.real)"""


