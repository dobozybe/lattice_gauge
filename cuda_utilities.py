import numpy as np
from numba import cuda,complex128
from cuda_kernels import *
import time




lie_gens = np.stack([ #i sigma_n where sigma_i are the paulis.
    np.array([[0, 1], [-1, 0]]),
    np.array([[0, 1j], [1j, 0]]),
    np.array([[1j, 0], [0, -1j]])
])


@cuda.jit(device=True, inline= True)
def link_update(config, dt, idx, lie_gens): #out should have same shape as in, so (n, d, 2,2)
    links = config[0]
    momentum = config[1]
    inshape = links.shape

    numnodes = inshape[0]
    numdims = inshape[1]

    total_matricies = numnodes * numdims

    if idx >= total_matricies:
        return

    nodeindex = idx//numdims
    direction = idx % numdims

    temp = cuda.local.array((2,2), dtype = complex128)

    scale_2x2_cuda(momentum[nodeindex, direction], dt, temp)
    #print("right before exponentiating! (commented)")
    su2_exp_cuda(temp, temp, lie_gens, idx)
    matmul_2x2_cuda(temp, links[nodeindex, direction], temp)

    for i in range(2):
        for j in range(2):
            config[0][nodeindex, direction, i,j] = temp[i,j]


@cuda.jit(device=True, inline= True)
def momentum_update(config, dt, staple_index_array, Barray, V2Barray, idx, out, g):

    links = config[0]
    momentum = config[1]
    inshape = links.shape

    numnodes = inshape[0]
    numdims = inshape[1]

    total_matricies = numnodes * numdims

    if idx >= total_matricies:
        return


    nodeindex = idx // numdims
    direction = idx % numdims


    if idx == 0:
        print("") #there for debug reasons, not entirely sure what's happening here

    #making the staple


    temp = cuda.local.array((2, 2), dtype=complex128)
    temp2 = cuda.local.array((2, 2), dtype=complex128)
    Vdirection = cuda.local.array((2,2), dtype=complex128)

    Vdirection[0,0] = 0
    Vdirection[0, 1] = 0
    Vdirection[1, 0] = 0
    Vdirection[1, 1] = 0

    for i in range(numdims):
        idx_tuple_1 = staple_index_array[nodeindex, direction,i, 0, 0]
        idx_tuple_2 = staple_index_array[nodeindex, direction, i, 0, 1]
        idx_tuple_3 = staple_index_array[nodeindex, direction, i, 0, 2]
        idx_tuple_4 = staple_index_array[nodeindex, direction, i, 1, 0]
        idx_tuple_5 = staple_index_array[nodeindex, direction, i, 1, 1]
        idx_tuple_6 = staple_index_array[nodeindex, direction, i, 1, 2]

        staple_matrix_1 = links[idx_tuple_1[0], idx_tuple_1[1]]
        staple_matrix_2 = links[idx_tuple_2[0], idx_tuple_2[1]]
        staple_matrix_3 = links[idx_tuple_3[0], idx_tuple_3[1]]

        staple_matrix_4 = links[idx_tuple_4[0], idx_tuple_4[1]]
        staple_matrix_5 = links[idx_tuple_5[0], idx_tuple_5[1]]
        staple_matrix_6 = links[idx_tuple_6[0], idx_tuple_6[1]]




        Bval = Barray[nodeindex, direction, i]
        V2Bval = V2Barray[nodeindex, direction, i]

        #calculating first staple
        dagger_2x2_cuda(staple_matrix_2, temp)

        matmul_2x2_cuda(staple_matrix_1, temp, temp2)

        dagger_2x2_cuda(staple_matrix_3, temp)



        matmul_2x2_cuda(temp2, temp, temp2)

        scale_2x2_cuda(temp2, Bval, temp2)
        add_2x2_cuda(temp2, Vdirection, Vdirection)

        # calculating second staple
        dagger_2x2_cuda(staple_matrix_4, temp)
        dagger_2x2_cuda(staple_matrix_5, temp2)
        matmul_2x2_cuda(temp, temp2, temp2)
        matmul_2x2_cuda(temp2, staple_matrix_6, temp2)
        scale_2x2_cuda(temp2, V2Bval, temp2)
        add_2x2_cuda(temp2, Vdirection, Vdirection)


    #calculating staple contribution
    matmul_2x2_cuda(links[nodeindex, direction], Vdirection, temp)

    dagger_2x2_cuda(temp, temp2)

    scale_2x2_cuda(temp, -1, temp)
    add_2x2_cuda(temp2, temp, temp)

    scale_2x2_cuda(temp, 1/g**2, temp)
    scale_2x2_cuda(temp, dt, temp)

    #calculating new momentum
    add_2x2_cuda(momentum[nodeindex, direction], temp, temp2)
    for i in range(2):
        for j in range(2):
            config[1][nodeindex, direction,i,j] = temp2[i,j]





@cuda.jit(device = True, inline = True)
def evolution_step(config, dt, staple_index_array, Barray, V2Barray, idx, g, lie_gens):
    momentum_update(config, dt, staple_index_array, Barray, V2Barray, idx, config, g)
    link_update(config, dt, idx, lie_gens, config)


@cuda.jit
def momentum_update_kernel(config, dt, staple_index_array_in, Barray_in, V2_Barray_in, g_in):
    idx = cuda.grid(1)
    numnodes = config[0].shape[0]
    numdims = config[0].shape[1]

    total_matricies = numnodes * numdims

    momentum_update(config, dt, staple_index_array_in, Barray_in, V2_Barray_in, idx, config, g_in)


@cuda.jit
def link_update_kernel(config, dt, lie_gens):
    idx = cuda.grid(1)
    numnodes = config[0].shape[0]


    numdims = config[0].shape[1]

    total_matricies = numnodes * numdims

    #print("started link")

    link_update(config, dt, idx, lie_gens)


def evolution_step_kernel(config, dt, staple_index_array, Barray, V2Barray, g, lie_gens):
    idx = cuda.grid(1)
    momentum_update(config, dt, staple_index_array, Barray, V2Barray, idx, config, g)
    link_update(config, dt, idx, lie_gens, config)


@cuda.jit
def time_evolve_kernel(config, dt, staple_index_array_in, Barray_in, V2_Barray_in, g_in, nsteps, lie_gens):
    idx = cuda.grid(1)
    numnodes = config[0].shape[0]
    numdims = config[0].shape[1]

    total_matricies = numnodes * numdims

    if idx >= total_matricies:
        return

    momentum_update(config, dt/2, staple_index_array_in, Barray_in, V2_Barray_in, idx, config, g_in)


    link_update(config, dt, idx, lie_gens, config)

    for i in range(nsteps-1):
        evolution_step(config, dt, staple_index_array_in, Barray_in, V2_Barray_in, idx, g_in, lie_gens)

    momentum_update(config, dt/2, staple_index_array_in, Barray_in, V2_Barray_in, idx, config, g_in)


def _parallel_time_evolve(initial_config, dt, staple_index_array_in, Barray_in, V2_Barray_in, g_in, processes, nsteps = 10000): #processes there for compatibility, ignore

    config = cuda.to_device(initial_config)
    staple_gpu = cuda.to_device(staple_index_array_in)
    Barray_gpu = cuda.to_device(np.squeeze(Barray_in))
    V2Barray_gpu= cuda.to_device(np.squeeze(V2_Barray_in))



    lie_gens = cuda.to_device(np.stack([ #i sigma_n where sigma_i are the paulis.
    np.array([[0, 1], [-1, 0]], dtype=np.complex128),
    np.array([[0, 1j], [1j, 0]],dtype=np.complex128),
    np.array([[1j, 0], [0, -1j]],dtype=np.complex128)]))


    configshape = initial_config[0].shape

    num_matricies = configshape[0]*configshape[1]

    threads_per_block = 64

    blocks = (num_matricies + threads_per_block - 1)//threads_per_block

    # Before launching kernels
    print("Free memory:", cuda.current_context().get_memory_info())

    # Calculate how much memory you're using
    numnodes = 24 * 6 * 6 * 24  # = 20736
    numdims = 4  # I assume?
    total_matricies = numnodes * numdims

    # Each link is a 2x2 complex matrix = 2*2*16 bytes = 64 bytes
    links_size = total_matricies * 64
    momentum_size = total_matricies * 64
    print(f"Links array: {links_size / 1e6:.1f} MB")
    print(f"Momentum array: {momentum_size / 1e6:.1f} MB")
    print(f"Staple index array: {staple_gpu.nbytes / 1e6:.1f} MB")
    print(f"Total: {(links_size + momentum_size + staple_gpu.nbytes) / 1e6:.1f} MB")

    print(f"Launching {blocks} blocks × {threads_per_block} threads = {blocks * threads_per_block} total threads")
    print(f"For {total_matricies} matrices")

    momentum_update_kernel[blocks, threads_per_block](config, dt / 2, staple_gpu, Barray_gpu, V2Barray_gpu, g_in)
    cuda.synchronize()
    print("momentum updated")



    link_update_kernel[blocks, threads_per_block](config, dt, lie_gens)

    cuda.synchronize()

    print("done initials")

    for i in range(nsteps-1):
        print(i)
        momentum_update_kernel[blocks, threads_per_block](config, dt, staple_gpu, Barray_gpu, V2Barray_gpu, g_in)
        cuda.synchronize()
        link_update_kernel[blocks, threads_per_block](config, dt, lie_gens)


    momentum_update_kernel[blocks, threads_per_block](config, dt/2, staple_gpu, Barray_gpu, V2Barray_gpu, g_in)

    return config.copy_to_host()