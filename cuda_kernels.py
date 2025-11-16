
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


@cuda.jit(device=True)
def reunitarize_su2_cuda(U):
    """
    Reunitarize a 2x2 SU(2) matrix in place.
    For SU(2), the second column is determined by the first column:
    U = [[a, -conj(c)],
         [c,  conj(a)]]
    where |a|^2 + |c|^2 = 1
    """
    # First normalize the first column
    norm_sq = (U[0, 0].real ** 2 + U[0, 0].imag ** 2 +
               U[1, 0].real ** 2 + U[1, 0].imag ** 2)
    norm = norm_sq ** 0.5

    if norm > 0:
        U[0, 0] = U[0, 0] / norm
        U[1, 0] = U[1, 0] / norm

    # Set second column based on first column
    # U[0,1] = -conj(U[1,0])
    # U[1,1] = conj(U[0,0])
    U[0, 1] = complex(-U[1, 0].real, U[1, 0].imag)
    U[1, 1] = complex(U[0, 0].real, -U[0, 0].imag)

    result00 = (U[0][0].real * U[0][0].real + U[0][0].imag * U[0][0].imag +
                U[0][1].real * U[0][1].real + U[0][1].imag * U[0][1].imag)
    result11 = (U[1][0].real * U[1][0].real + U[1][0].imag * U[1][0].imag +
                U[1][1].real * U[1][1].real + U[1][1].imag * U[1][1].imag)

    # Crash if not unitary (diagonal elements should be 1)
    if abs(result00 - 1.0) > 1e-6 or abs(result11 - 1.0) > 1e-6:
        cuda.syncthreads()
        U[0][0] = complex(1.0 / 0.0, 0.0)  # Division by zero to crash


@cuda.jit(device=True, inline=True)
def extra_term_derivative(action):
    a = 0.1
    S = 39.48
    scale = 1
    prefactor = 1 / (math.sqrt(math.pi) * a)
    return 0
    return -2 * scale * prefactor * (
            (action - S + 0.1 * S) * math.exp(-(action - S + 0.1 * S) ** 2 / a ** 2) + (
                action - S - 0.1 * S) * math.exp(-(action - S - 0.1 * S) ** 2 / a ** 2)
    )


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

@cuda.jit(device=True)
def get_node_direction_action_contrib(idx, config, Barray, staple_index_array):
    links = config[0]
    inshape = links.shape

    numnodes = inshape[0]
    numdims = inshape[1]

    total_matricies = numnodes * numdims

    if idx >= total_matricies:
        return


    nodeindex = idx // numdims
    direction = idx % numdims

    #making the staple

    tracecount = 0
    temp = cuda.local.array((2, 2), dtype=complex128)
    temp2 = cuda.local.array((2, 2), dtype=complex128)

    this_matrix = links[nodeindex, direction]

    for i in range(numdims):
        if i != direction:
            idx_tuple_1 = staple_index_array[nodeindex, direction,i, 0, 0]
            idx_tuple_2 = staple_index_array[nodeindex, direction, i, 0, 1]
            idx_tuple_3 = staple_index_array[nodeindex, direction, i, 0, 2]


            staple_matrix_1 = links[idx_tuple_1[0], idx_tuple_1[1]]
            staple_matrix_2 = links[idx_tuple_2[0], idx_tuple_2[1]]
            staple_matrix_3 = links[idx_tuple_3[0], idx_tuple_3[1]]

            if staple_matrix_1[0][0] != staple_matrix_1[0][0]:
                print("=== NaN in staple_matrix_1 at idx", idx, "===")
            if staple_matrix_2[0][0] != staple_matrix_2[0][0]:
                print("=== NaN in staple_matrix_2 at idx", idx, "===")
            if staple_matrix_3[0][0] != staple_matrix_3[0][0]:
                print("=== NaN in staple_matrix_3 at idx", idx, "===")

            Bval = Barray[nodeindex, direction, i]

            matmul_2x2_cuda(this_matrix, staple_matrix_1, temp)

            dagger_2x2_cuda(staple_matrix_2, temp2)

            matmul_2x2_cuda(temp, temp2, temp)

            dagger_2x2_cuda(staple_matrix_3, temp2)

            matmul_2x2_cuda(temp, temp2, temp) #computed the [direction, i] plaquette holonomy

            scale_2x2_cuda(temp, Bval, temp) #scaled by B[direction, i]

            tracecount+=trace_2x2_cuda(temp)





    """if tracecount != tracecount:
        print(idx, "bval", Bval.real)
        print("temp")
        print(temp[0][0].real, temp[0][0].imag)
        print(temp[0][1].real, temp[0][1].imag)
        print(temp[1][0].real, temp[1][0].imag)
        print(temp[1][1].real, temp[1][1].imag)"""



    return tracecount



@cuda.jit(device=True, inline = True)
def polyakov_loop(basenodeindex,links, index_increment, time_length, out):
    outarray = cuda.local.array((2,2), dtype = complex128)
    startmatrix = links[basenodeindex,0]
    outarray[0][0] = startmatrix[0][0]
    outarray[0][1] = startmatrix[0][1]
    outarray[1][0] = startmatrix[1][0]
    outarray[1][1] = startmatrix[1][1]


    for i in range(1, time_length):
        nodeindex = (basenodeindex + index_increment * i) % (index_increment * time_length)
        link_matrix = links[nodeindex, 0]



        # Check unitarity: U * U^dagger diagonal elements should be 1
        norm0 = link_matrix[0][0].real ** 2 + link_matrix[0][0].imag ** 2 + link_matrix[0][1].real ** 2 + \
                link_matrix[0][1].imag ** 2
        norm1 = link_matrix[1][0].real ** 2 + link_matrix[1][0].imag ** 2 + link_matrix[1][1].real ** 2 + \
                link_matrix[1][1].imag ** 2

        if abs(norm0 - 1.0) > 1e-6 or abs(norm1 - 1.0) > 1e-6:
            print("=== Non-unitary link at node", nodeindex, "bnindex", basenodeindex, "i=", i, "norms:", norm0, norm1,
                  "===")
            print("Link:", link_matrix[0][0].real, link_matrix[0][0].imag, link_matrix[0][1].real,
                  link_matrix[0][1].imag, link_matrix[1][0].real, link_matrix[1][0].imag, link_matrix[1][1].real,
                  link_matrix[1][1].imag)


        matmul_2x2_cuda(outarray, link_matrix, outarray)

    mag = (outarray[0][0].real ** 2 + outarray[0][0].imag ** 2) ** 0.5
    if mag > 1e10:
        print("=== Polyakov loop overflow at node", basenodeindex, "mag =", mag, "===")

    out[0][0] = outarray[0][0]
    out[0][1] = outarray[0][1]
    out[1][0] = outarray[1][0]
    out[1][1] = outarray[1][1]