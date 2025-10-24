import os

import utilities


from utilities import *
import multiprocessing as mp
import multiprocessing.shared_memory as shared_memory
import numpy as np
import time
from numba import *
set_num_threads(8)

momentum_time_list = []
link_time_list =[]

@njit(parallel=True)
def matmul_chain(A, B, C):
    """Computes A@B@C for arrays of 2x2 matrices
    Input shape: (n, 4, 4, 2, 2)
    Output shape: (n, 4, 4, 2, 2)
    """
    n = A.shape[0]
    out_array = np.empty((n, 4, 4, 2, 2), dtype=np.complex128)

    for i in prange(n):  # Parallelize outer loop
        for mu in range(4):
            for nu in range(4):
                # Extract 2x2 matrices
                mat_A = A[i, mu, nu]
                mat_B = B[i, mu, nu]
                mat_C = C[i, mu, nu]

                # Compute A @ B manually (unrolled for 2x2)
                temp00 = mat_A[0, 0] * mat_B[0, 0] + mat_A[0, 1] * mat_B[1, 0]
                temp01 = mat_A[0, 0] * mat_B[0, 1] + mat_A[0, 1] * mat_B[1, 1]
                temp10 = mat_A[1, 0] * mat_B[0, 0] + mat_A[1, 1] * mat_B[1, 0]
                temp11 = mat_A[1, 0] * mat_B[0, 1] + mat_A[1, 1] * mat_B[1, 1]

                # Compute temp @ C manually
                out_array[i, mu, nu,0,0] = temp00 * mat_C[0, 0] + temp01 * mat_C[1, 0]
                out_array[i, mu, nu,0, 1] = temp00 * mat_C[0, 1] + temp01 * mat_C[1, 1]
                out_array[i, mu, nu, 1, 0] = temp10 * mat_C[0, 0] + temp11 * mat_C[1, 0]
                out_array[i, mu, nu, 1, 1] = temp10 * mat_C[0, 1] + temp11 * mat_C[1, 1]

    return out_array





@njit(parallel=True)
def _make_staple_array(staple_index_array, link_array):
    array_shape = np.shape(staple_index_array)
    staple_matrix_array = np.zeros(array_shape+ (2,), dtype=np.complex128)

    for nodeindex in prange(array_shape[0]):
        for mu in range(array_shape[1]):
            for nu in range(array_shape[2]):
                if mu!=nu:
                    for term in range(2):
                        for staple_index in range(3):
                            linkarray_index = staple_index_array[nodeindex, mu, nu, term, staple_index]
                            matrix = link_array[linkarray_index[0]][linkarray_index[1]]
                            staple_matrix_array[nodeindex, mu, nu, term, staple_index,0,0] = matrix[0,0]
                            staple_matrix_array[nodeindex, mu, nu, term, staple_index, 0, 1] = matrix[0, 1]
                            staple_matrix_array[nodeindex, mu, nu, term, staple_index, 1, 0] = matrix[1, 0]
                            staple_matrix_array[nodeindex, mu, nu, term, staple_index, 1, 1] = matrix[1, 1]
    return staple_matrix_array

@njit(parallel=True)
def _make_staple_lists(staple_matrix_array):
    array_shape = staple_matrix_array.shape[0:3]
    staple_matricies = staple_matrix_array

    staple11 = np.zeros((array_shape) + (2, 2), dtype=np.complex128)
    staple12 = np.zeros((array_shape) + (2, 2), dtype=np.complex128)
    staple13 = np.zeros((array_shape) + (2, 2), dtype=np.complex128)
    staple21 = np.zeros((array_shape) + (2, 2), dtype=np.complex128)
    staple22 = np.zeros((array_shape) + (2, 2), dtype=np.complex128)
    staple23 = np.zeros((array_shape) + (2, 2), dtype=np.complex128)


    for nodeindex in prange(array_shape[0]):
        for mu in range(array_shape[1]):
            for nu in range(array_shape[1]):
                Bval = Barray[nodeindex, mu, nu][0][0]
                V2Bval = V2_Barray[nodeindex, mu, nu][0][0]
                staple11[nodeindex, mu, nu, 0,0] = Bval * staple_matricies[nodeindex, mu, nu, 0, 0, 0, 0]
                staple11[nodeindex, mu, nu, 0, 1] = Bval * staple_matricies[nodeindex, mu, nu, 0, 0, 0, 1]
                staple11[nodeindex, mu, nu, 1, 0] = Bval * staple_matricies[nodeindex, mu, nu, 0, 0, 1, 0]
                staple11[nodeindex, mu, nu, 1, 1] = Bval * staple_matricies[nodeindex, mu, nu, 0, 0, 1, 1]

                staple12[nodeindex, mu, nu, 0, 0] = np.conj(staple_matricies[nodeindex, mu, nu, 0, 1, 0, 0])
                staple12[nodeindex, mu, nu, 0, 1] = np.conj(staple_matricies[nodeindex, mu, nu, 0, 1, 1,0])
                staple12[nodeindex, mu, nu, 1, 0] = np.conj(staple_matricies[nodeindex, mu, nu, 0, 1, 0,1])
                staple12[nodeindex, mu, nu, 1, 1] = np.conj(staple_matricies[nodeindex, mu, nu, 0, 1, 1, 1])

                staple13[nodeindex, mu, nu, 0, 0] = np.conj(staple_matricies[nodeindex, mu, nu, 0, 2, 0, 0])
                staple13[nodeindex, mu, nu, 0, 1] = np.conj(staple_matricies[nodeindex, mu, nu, 0, 2, 1,0])
                staple13[nodeindex, mu, nu, 1, 0] = np.conj(staple_matricies[nodeindex, mu, nu, 0, 2, 0,1])
                staple13[nodeindex, mu, nu, 1, 1] = np.conj(staple_matricies[nodeindex, mu, nu, 0, 2, 1, 1])

                staple21[nodeindex, mu, nu, 0, 0] = V2Bval * np.conj(staple_matricies[nodeindex, mu, nu, 1, 0, 0, 0])
                staple21[nodeindex, mu, nu, 0, 1] = V2Bval * np.conj(staple_matricies[nodeindex, mu, nu, 1, 0, 1, 0])
                staple21[nodeindex, mu, nu, 1, 0] = V2Bval * np.conj(staple_matricies[nodeindex, mu, nu, 1, 0, 0, 1])
                staple21[nodeindex, mu, nu, 1, 1] = V2Bval * np.conj(staple_matricies[nodeindex, mu, nu, 1, 0, 1, 1])

                staple22[nodeindex, mu, nu, 0, 0] = np.conj(staple_matricies[nodeindex, mu, nu, 1, 1, 0, 0])
                staple22[nodeindex, mu, nu, 0, 1] = np.conj(staple_matricies[nodeindex, mu, nu, 1, 1, 1, 0])
                staple22[nodeindex, mu, nu, 1, 0] = np.conj(staple_matricies[nodeindex, mu, nu, 1, 1, 0, 1])
                staple22[nodeindex, mu, nu, 1, 1] = np.conj(staple_matricies[nodeindex, mu, nu, 1, 1, 1, 1])

                staple23[nodeindex, mu, nu, 0, 0] = staple_matricies[nodeindex, mu, nu, 1, 2, 0, 0]
                staple23[nodeindex, mu, nu, 0, 1] = staple_matricies[nodeindex, mu, nu, 1, 2, 0,1]
                staple23[nodeindex, mu, nu, 1, 0] = staple_matricies[nodeindex, mu, nu, 1, 2, 1,0]
                staple23[nodeindex, mu, nu, 1, 1] = staple_matricies[nodeindex, mu, nu, 1, 2, 1,1]

    return staple11, staple12, staple13, staple21, staple22, staple23


@njit(parallel=True)
def two_matmul_chain(A,B): #input shapes: (n, dimensions, 2,2). Same as output.
    inshape = A.shape
    outarray = np.zeros(inshape, dtype = np.complex128)
    for nodeindex in prange(inshape[0]):
        for mu in range(inshape[1]):
            for i in range(2):
                for k in range(2):
                    for j in range(2):
                        outarray[nodeindex,mu,i,k] += A[nodeindex,mu, i,j] * B[nodeindex,mu,j,k]
    return outarray


@njit(parallel=True)
def get_momentum_change(stapleterm, dt):
    inshape = stapleterm.shape
    outarray = np.zeros(inshape, dtype=np.complex128)

    for nodeindex in prange(inshape[0]):
        for direction in range(inshape[1]):
            for i in range(2):
                for j in range(2):
                    outarray[nodeindex, direction,i,j] = (1/g**2) * ((np.conj(stapleterm[nodeindex, direction,j,i]) - stapleterm[nodeindex,direction,i,j]) * dt)
    return outarray


@njit(parallel=True)
def momentum_update(dt, g, action, config): #ND numpy array nodecoords
    #starttime = time.time()
    link_array = config[0]
    momentum_array = config[1]
    staple_matricies = _make_staple_array(staple_index_array, config[0])


    # indicies are node, direction mu, direction nu, first/second term, list of staple matricies

    staple11, staple12, staple13, staple21, staple22, staple23 = _make_staple_lists(staple_matricies)



    # calculate the twisted staple for each nu (index 2)
    #firstsegtime = time.time()-starttime


    firststaple = matmul_chain(staple11, staple12, staple13)
    secondstaple =matmul_chain(staple21, staple22, staple23)
    stapleshape = firststaple.shape
    #starttime = time.time()

    # adding two terms together to get full staple then summing to get Vmu array. Squeeze to get rid of vestigal indicies

    Varray = np.zeros((stapleshape[0], stapleshape[1],2,2), dtype = np.complex128)

    for nodeindex in prange(stapleshape[0]):
        for mu in range(stapleshape[1]):
            for nu in range(stapleshape[1]):
                for i in range(2):
                    for j in range(2):
                        Varray[nodeindex,mu,i,j] += firststaple[nodeindex,mu,nu,i,j] + secondstaple[nodeindex,mu,nu,i,j]

    # calculating momentum update
    stapleterm = two_matmul_chain(link_array, Varray)
    momentum_change = get_momentum_change(stapleterm, dt)

    momentumshape = momentum_array.shape
    new_momentum_array = np.zeros(momentumshape, dtype = np.complex128)

    for nodeindex in prange(momentumshape[0]):
        for direction in range(momentumshape[1]):
            for i in range(2):
                for j in range(2):
                    new_momentum_array[nodeindex, direction, i,j] = momentum_array[nodeindex,direction, i,j] + momentum_change[nodeindex, direction, i,j]


    #lastseg = time.time()-starttime
    #print("time other than mult", firstsegtime + lastseg)
    return new_momentum_array



def get_wilson_action(configuration = None, staple_index = None, barray = None): #assumes g=1
    global config
    global staple_matrix_array
    if configuration is not None and staple_index is not None and barray is not None:
        link_matricies = configuration[0]
        staple_matricies = _make_staple_array(staple_index, link_matricies)
        momentum_array = configuration[1]
        Barray = barray
    else:
        link_matricies = config[0]
        staple_matricies = staple_matrix_array
    firststaplematricies, secondstaplematricies = np.split(staple_matricies, 2, axis=3)

    staple11, staple12, staple13 = np.split(firststaplematricies, 3, axis=4)

    firststaple = Barray[..., None, None] * staple11 @ staple12.conj().swapaxes(-1,-2) @ staple13.conj().swapaxes(-1, -2)

    firststaple = np.sum(firststaple, axis = 2).squeeze()


    sum_nu_plaquettes = link_matricies @ firststaple

    holonomy = np.sum(np.trace(sum_nu_plaquettes, axis1 = -1, axis2 = -2), axis = (0,1))

    shape = np.shape(link_matricies)

    action = 2 * shape[0] * shape[1] * (shape[1]-1) - holonomy

    daggered_momentum = momentum_array.conj().transpose(0, 1, 3, 2)

    momentum_contribution = 0.5 * np.sum(np.trace(momentum_array @ daggered_momentum, axis1=2, axis2=3), axis=(0, 1))
    """print("momentum", momentum_contribution)
    print("wilson action", action)
    print("fictitious action", extra_action_term(action))
    print("total action", action + extra_action_term_derivative(action))
    print("hamiltonian", action + extra_action_term_derivative(action) + momentum_contribution)"""
    return action





@njit(parallel=True)
def link_update(dt, config):

        link_array = config[0]
        momentum_array = config[1]

        inshape = momentum_array.shape

        scaled_momentum = np.zeros(inshape, dtype=np.complex128)
        for nodeindex in prange(inshape[0]):
            for direction in range(inshape[1]):
                scaled_momentum[nodeindex, direction] = jit_2x2_scale(momentum_array[nodeindex,direction], dt)
        exp_out = jit_su2_exp(scaled_momentum)

        updated_out = np.zeros(inshape, dtype= np.complex128)
        for nodeindex in prange(inshape[0]):
            for direction in range(inshape[1]):
                updated_out[nodeindex,direction] = jit_2x2_mult(exp_out[nodeindex,direction], link_array[nodeindex,direction])

        return updated_out


def _link_update(dt, config):

    link_array = config[0]
    momentum_array = config[1]

    batch_link = link_array
    batch_momentum = momentum_array

    batch_out = su2_exp(dt * batch_momentum) @ batch_link
    return batch_out


def parallel_evolution_step(config, dt):
    #print('starting evolution step')

    start =time.time()
    action = (1/g**2) * get_wilson_action(configuration=config, staple_index=staple_index_array, barray=Barray)
    #print("Action in", time.time()-start)
    start = time.time()
    new_momentum_array = momentum_update(dt, g, action, config)
    config[:] = np.stack([config[0], new_momentum_array])
    momentum_time_list.append(time.time()-start)

    start = time.time()
    new_link_array = link_update(dt, config)

    link_time_list.append(time.time()-start)


    start = time.time()
    config[:] = np.stack([new_link_array, new_momentum_array])
    #print("stacked!", time.time()-start)

    start = time.time()
    return config
    #print("staple matrix array updated", time.time()-start)














def _parallel_time_evolve(initial_config, dt, staple_index_array_in, Barray_in, V2_Barray_in, g_in, processes, nsteps=10000):
    overall = time.time()
    global Barray
    global V2_Barray
    global staple_index_array
    global g

    num_nodes = np.shape(Barray_in)[0]


    Barray = Barray_in
    V2_Barray=V2_Barray_in
    staple_index_array = staple_index_array_in
    g = g_in



    array_shape = np.shape(initial_config[0])

    config = initial_config.copy()




    #print("first momentum update")
    momen_start = time.time()
    action = (1/g**2) * get_wilson_action(configuration=config, staple_index=staple_index_array, barray = Barray)
    print("first action", time.time()-momen_start)
    new_momentum_array = momentum_update(dt/2, g, action, config)
    config[1] = new_momentum_array
    #print("end of first momentum update")



    link_start = time.time()
    new_link_array = link_update(dt, config)
    config[0] = new_link_array


    next_action = time.time()
    action2 = (1 / g ** 2) * get_wilson_action(configuration=config, staple_index=staple_index_array, barray=Barray)
    print("next action", time.time()-next_action)
    #print("action change after first two steps", action2-action)
    start = time.time()
    for i in range(nsteps - 1):
        #print("step", i)
        parallel_evolution_step(config, dt)
    print("time for evolution steps", time.time()-start)
    print("average momentum time:", np.average(momentum_time_list))
    print("average link time:", np.average(link_time_list))
    #print("last action in time evolve")

    last_action = time.time()
    action = (1 / g ** 2) * get_wilson_action(configuration=config, staple_index=staple_index_array, barray=Barray)
    print("last action", time.time()-last_action)

    new_momentum_array = momentum_update(dt/2, g, action, config)
    config[1] = new_momentum_array

    print("internal time evolve", time.time()-overall)
    return config



