import numpy as np
from numba import cuda,complex128
from cuda_kernels import *
import time

import warnings
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)


lie_gens = np.stack([ #i sigma_n where sigma_i are the paulis.
    np.array([[0, 1], [-1, 0]]),
    np.array([[0, 1j], [1j, 0]]),
    np.array([[1j, 0], [0, -1j]])
])


"""@cuda.jit(device=True, inline= True)
def link_update(config, dt, idx, lie_gens): #out should have same shape as in, so (n, d, 2,2)
    links = config[0]
    momentum = config[1]
    inshape = links.shape

    numnodes = inshape[0]
    numdims = inshape[1]

    total_matricies = numnodes * numdims

    if idx >= total_matricies:
        return

    if idx == 3744:


    nodeindex = idx//numdims
    direction = idx % numdims

    temp = cuda.local.array((2,2), dtype = complex128)

    scale_2x2_cuda(momentum[nodeindex, direction], dt, temp)
    #print("right before exponentiating! (commented)")
    su2_exp_cuda(temp, temp, lie_gens, idx)
    matmul_2x2_cuda(temp, links[nodeindex, direction], temp)

    reunitarize_su2_cuda(temp)

    for i in range(2):
        for j in range(2):
            config[0][nodeindex, direction, i,j] = temp[i,j]"""


@cuda.jit(device=True, inline=True)
def link_update(config, dt, idx, lie_gens):  # out should have same shape as in, so (n, d, 2,2)
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

    temp = cuda.local.array((2, 2), dtype=complex128)

    # Check for NaN in momentum
    if (momentum[nodeindex, direction][0][0] != momentum[nodeindex, direction][0][0]):
        print("=== NaN in momentum at idx", idx, "node", nodeindex, "dir", direction, "===")

    scale_2x2_cuda(momentum[nodeindex, direction], dt, temp)

    # Check after scaling
    if temp[0][0] != temp[0][0]:
        print("=== NaN after scale at idx", idx, "===")
        print("mom:", momentum[nodeindex, direction][0][0].real, momentum[nodeindex, direction][0][0].imag,
              momentum[nodeindex, direction][0][1].real, momentum[nodeindex, direction][0][1].imag,
              momentum[nodeindex, direction][1][0].real, momentum[nodeindex, direction][1][0].imag,
              momentum[nodeindex, direction][1][1].real, momentum[nodeindex, direction][1][1].imag)
        print("dt:", dt)

    su2_exp_cuda(temp, temp, lie_gens, idx)

    # Check after exponential
    if temp[0][0] != temp[0][0]:
        print("=== NaN after exp at idx", idx, "===")
        print("temp:", temp[0][0].real, temp[0][0].imag, temp[0][1].real, temp[0][1].imag, temp[1][0].real,
              temp[1][0].imag, temp[1][1].real, temp[1][1].imag)

    # Check link before matmul
    if links[nodeindex, direction][0][0] != links[nodeindex, direction][0][0]:
        print("=== NaN in link before matmul at idx", idx, "===")

    matmul_2x2_cuda(temp, links[nodeindex, direction], temp)

    # Check after matmul
    if temp[0][0] != temp[0][0]:
        print("=== NaN after matmul at idx", idx, "===")
        print("link:", links[nodeindex, direction][0][0].real, links[nodeindex, direction][0][0].imag,
              links[nodeindex, direction][0][1].real, links[nodeindex, direction][0][1].imag,
              links[nodeindex, direction][1][0].real, links[nodeindex, direction][1][0].imag,
              links[nodeindex, direction][1][1].real, links[nodeindex, direction][1][1].imag)

    reunitarize_su2_cuda(temp)

    # Check after reunitarize
    if temp[0][0] != temp[0][0]:
        print("=== NaN after reunitarize at idx", idx, "===")

    result00 = (temp[0][0].real * temp[0][0].real + temp[0][0].imag * temp[0][0].imag +
                temp[0][1].real * temp[0][1].real + temp[0][1].imag * temp[0][1].imag)
    result11 = (temp[1][0].real * temp[1][0].real + temp[1][0].imag * temp[1][0].imag +
                temp[1][1].real * temp[1][1].real + temp[1][1].imag * temp[1][1].imag)

    # Crash if not unitary (diagonal elements should be 1)
    if abs(result00 - 1.0) > 1e-6 or abs(result11 - 1.0) > 1e-6:
        temp[0][0] = 1 / 0  # Division by zero to crash

    for i in range(2):
        for j in range(2):
            config[0][nodeindex, direction, i, j] = temp[i, j]


@cuda.jit(device=True, inline= True)
def momentum_update(config, dt, staple_index_array, Barray, V2Barray, idx, g, action, deformation, scaling_param, deformation_coeff):

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


    # Check initial momentum
    """if momentum[nodeindex, direction][0][0] != momentum[nodeindex, direction][0][0]:
        print("=== NaN in initial momentum at idx", idx, "===")"""

    #making the staple


    temp = cuda.local.array((2, 2), dtype=complex128)
    temp2 = cuda.local.array((2, 2), dtype=complex128)
    Vdirection = cuda.local.array((2,2), dtype=complex128)

    Vdirection[0,0] = 0
    Vdirection[0, 1] = 0
    Vdirection[1, 0] = 0
    Vdirection[1, 1] = 0


    for i in range(numdims):
        if i != direction:
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

    """if temp[0][0] != temp[0][0]:
        print("=== NaN after matmul with Vdirection at idx", idx, "===")"""

    dagger_2x2_cuda(temp, temp2)

    scale_2x2_cuda(temp, -1, temp)
    add_2x2_cuda(temp2, temp, temp)

    scale_2x2_cuda(temp, 1/g**2, temp)
    scale_2x2_cuda(temp, dt, temp)

    """if temp[0][0] != temp[0][0]:
        print("=== NaN after staple scaling at idx", idx, "===")"""

    # momentum change stored as temp



    #scale_factor = 1 + extra_term_derivative(action)

    #scale_2x2_cuda(temp, scale_factor, temp)

    if deformation[0] == 1 and direction==0:
        additional_term = cuda.local.array((2,2), dtype = complex128)
        polyakov_loop(nodeindex, links, deformation[1], deformation[2], additional_term)


        dagger_2x2_cuda(additional_term, temp2)
        daggerlooptrace = trace_2x2_cuda(temp2)

        """if daggerlooptrace != daggerlooptrace:
            print("=== NaN in daggerlooptrace at idx", idx, "===")"""

        scale_2x2_cuda(additional_term, daggerlooptrace,additional_term)

        """if additional_term[0][0] != additional_term[0][0]:
            print("=== NaN after scaling by daggerlooptrace at idx", idx, "===")"""

        #add the non-daggered term

        dagger_2x2_cuda(additional_term, temp2) #subtract the daggered term
        scale_2x2_cuda(temp2, -1, temp2)
        add_2x2_cuda(temp2, additional_term, additional_term)

        """if additional_term[0][0] != additional_term[0][0]:
            print("=== NaN in additional_term after dagger subtraction at idx", idx, "daggerlooptrace of", daggerlooptrace.real,"===")"""

        # scale by the coeff on the deformation potential and dt

        scale_2x2_cuda(additional_term, deformation_coeff, additional_term)
        scale_2x2_cuda(additional_term, dt, additional_term)

        add_2x2_cuda(temp, additional_term, temp)




    #calculating new momentum. First scale by scaling param

    scale_2x2_cuda(temp, scaling_param, temp)

    add_2x2_cuda(momentum[nodeindex, direction], temp, temp2)

    """if temp2[0][0] != temp2[0][0]:
        print("=== NaN in final momentum update at idx", idx, "===")"""

    for i in range(2):
        for j in range(2):
            config[1][nodeindex, direction,i,j] = temp2[i,j]



@cuda.jit(device = True, inline=True)
def get_wilson_action(config, staple_index_array, Barray, g, idx):

    links = config[0]
    inshape = links.shape

    numnodes = inshape[0]
    numdims = inshape[1]

    total_matricies = numnodes * numdims

    if idx >= total_matricies:
        return



    plaquette_contribs=get_node_direction_action_contrib(idx, config, Barray, staple_index_array)

    total_contrib = (1/g**2) * (2 * (numdims-1) - plaquette_contribs)


    return total_contrib


@cuda.jit(device = True, inline = True)
def get_deformation_contrib(config, deformation, idx):

    index_increment = deformation[1]
    links = config[0]

    if idx <index_increment:
        this_loop_matrix = cuda.local.array((2,2),dtype=complex128)
        polyakov_loop(idx, links, deformation[1], deformation[2], this_loop_matrix)
        looptrace = trace_2x2_cuda(this_loop_matrix)
        this_contrib = abs(looptrace**2)
        return this_contrib
    else:
        return 0



@cuda.jit
def momentum_update_kernel(config, dt, staple_index_array_in, Barray_in, V2_Barray_in, g_in, action_in, deformation, scaling_param, deformation_coeff):
    idx = cuda.grid(1)

    momentum_update(config, dt, staple_index_array_in, Barray_in, V2_Barray_in, idx, g_in, action_in, deformation, scaling_param, deformation_coeff)


@cuda.jit
def link_update_kernel(config, dt, lie_gens):
    idx = cuda.grid(1)

    #print("started link")

    link_update(config, dt, idx, lie_gens)

@cuda.jit
def wilson_action_kernel(config, staple_index_array, Barray, g, scaling_param, out):
    idx = cuda.grid(1)

    action_val = get_wilson_action(config, staple_index_array, Barray, g, idx)

    if action_val is not None:  # Check if valid index
        out[idx] = action_val.real * scaling_param  # Store real part
    else:
        pass

@cuda.jit
def deformation_potential_kernel(config, deformation, scaling_param, coeff, out): #deformation is a [time length, index increment] list
    if deformation[0]!=0:
        idx = cuda.grid(1)

        action_val = get_deformation_contrib(config, deformation, idx)

        out[idx] = action_val * scaling_param * coeff


def check_unitarity(config):
    links = config[0]
    numnodes, numdims = links.shape[0], links.shape[1]

    non_unitary_count = 0
    for nodeindex in range(numnodes):
        for direction in range(numdims):
            link = links[nodeindex, direction]
            norm0 = np.abs(link[0, 0]) ** 2 + np.abs(link[0, 1]) ** 2
            norm1 = np.abs(link[1, 0]) ** 2 + np.abs(link[1, 1]) ** 2

            if abs(norm0 - 1.0) > 1e-6 or abs(norm1 - 1.0) > 1e-6:
                print(f"Non-unitary at node {nodeindex}, dir {direction}: norms = {norm0}, {norm1}")
                print(f"  Link: {link}")
                non_unitary_count += 1

    if non_unitary_count == 0:
        print("All links are unitary!")
    else:
        print(f"Found {non_unitary_count} non-unitary links")












def _parallel_time_evolve(initial_config, dt, staple_index_array_in, Barray_in, V2_Barray_in, g_in, processes, nsteps = 10000, scaling_param = 1, deformation_data = None): #processes there for compatibility, ignore
    #print("time evolve sees scaling param of,", scaling_param)

    deformation = [0,0]
    deformation_coeff = 0
    if deformation_data != None:
        deformation = np.array(deformation_data[0]) #this is the list of shape data
        deformation_coeff = deformation_data[1]

        deformation = np.array(deformation)

    #check_unitarity(initial_config)


    config = cuda.to_device(initial_config)
    staple_gpu = cuda.to_device(staple_index_array_in)
    Barray_gpu = cuda.to_device(np.squeeze(Barray_in))
    V2Barray_gpu= cuda.to_device(np.squeeze(V2_Barray_in))
    deformation = cuda.to_device(deformation)


    lie_gens = cuda.to_device(np.stack([ #i sigma_n where sigma_i are the paulis.
    np.array([[0, 1], [-1, 0]], dtype=np.complex128),
    np.array([[0, 1j], [1j, 0]],dtype=np.complex128),
    np.array([[1j, 0], [0, -1j]],dtype=np.complex128)]))


    configshape = initial_config[0].shape

    num_matricies = configshape[0]*configshape[1]

    threads_per_block = 64

    blocks = (num_matricies + threads_per_block - 1)//threads_per_block

    action_holder = np.zeros(num_matricies, dtype=np.complex128)
    deformed_action_holder = np.zeros(num_matricies, dtype=np.complex128)

    gpu_action = cuda.to_device(action_holder)
    gpu_deformation = cuda.to_device(deformed_action_holder)


    deformation_potential_kernel[blocks, threads_per_block](config, deformation, scaling_param, deformation_coeff, gpu_deformation)
    deformation_contrib = gpu_deformation.copy_to_host().sum()

    wilson_action_kernel[blocks, threads_per_block](config, staple_gpu, Barray_gpu, g_in, scaling_param, gpu_action)
    action=gpu_action.copy_to_host().sum()





    #print("Action before", action + deformation_contrib, action, "/", deformation_contrib)

    momentum_update_kernel[blocks, threads_per_block](config, dt / 2, staple_gpu, Barray_gpu, V2Barray_gpu, g_in, action,deformation, scaling_param, deformation_coeff)


    link_update_kernel[blocks, threads_per_block](config, dt, lie_gens)


    for i in range(nsteps-1):
        print(i, end = "\r")

        #wilson_action_kernel[blocks, threads_per_block](config, staple_gpu, Barray_gpu, g_in, gpu_action)
        #action = gpu_action.copy_to_host().sum()

        momentum_update_kernel[blocks, threads_per_block](config, dt, staple_gpu, Barray_gpu, V2Barray_gpu, g_in, action,deformation, scaling_param, deformation_coeff)
        link_update_kernel[blocks, threads_per_block](config, dt, lie_gens)

    #wilson_action_kernel[blocks, threads_per_block](config, staple_gpu, Barray_gpu, g_in, gpu_action)
    #action = gpu_action.copy_to_host().sum()
    momentum_update_kernel[blocks, threads_per_block](config, dt/2, staple_gpu, Barray_gpu, V2Barray_gpu, g_in, action, deformation, scaling_param, deformation_coeff)

    wilson_action_kernel[blocks, threads_per_block](config, staple_gpu, Barray_gpu, g_in, scaling_param, gpu_action)
    action = gpu_action.copy_to_host().sum()

    deformation_potential_kernel[blocks, threads_per_block](config, deformation, scaling_param, deformation_coeff, gpu_deformation)
    deformation_contrib = gpu_deformation.copy_to_host().sum()


    afteraction = action + deformation_contrib
    #print("Action after", afteraction,action, "/", deformation_contrib)

    return config.copy_to_host()








