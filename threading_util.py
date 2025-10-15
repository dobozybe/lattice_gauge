import os
"""os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
"""
from utilities import *
import multiprocessing.dummy as mp

import numpy as np
import time




def _make_staple_array(staple_index_array, link_array):
    staple_matrix_array = np.zeros(np.shape(staple_index_array) + (2,), dtype=np.complex128)

    # Create a mask for where index[1] != index[2] (non-diagonal elements)
    mask = np.ones(staple_index_array.shape[:3], dtype=bool)
    diag_indices = np.arange(min(staple_index_array.shape[1], staple_index_array.shape[2]))
    mask[:, diag_indices, diag_indices] = False

    # Use advanced indexing to get all values at once
    # staple_index_array[mask] has shape (N*4*3, 2, 3, 2)
    # After transpose: (2, 3, 2, N*4*3)
    # link_array indexing gives: (3, 2, N*4*3, 2, 2)
    result = link_array[tuple(staple_index_array[mask].T)]

    # Rearrange dimensions to match staple_matrix_array[mask] shape: (N*4*3, 2, 3, 2, 2)
    result_reordered = result.transpose(2, 1, 0, 3, 4)

    staple_matrix_array[mask] = result_reordered

    return staple_matrix_array


def batched_single_node_momentum_update(dt, start,end, g): #ND numpy array nodecoords
    global shared_staple_matricies
    global shared_config

    config = shared_config
    staple_matrix_array = shared_staple_matricies

    link_array = config[0][start:end + 1]
    momentum_array = config[1][start:end + 1]
    staple_matricies = staple_matrix_array[start:end + 1]



    # indicies are node, direction mu, direction nu, first/second term, list of staple matricies

    # extract staple matricies

    firststaplematricies, secondstaplematricies = np.split(staple_matricies, 2, axis=3)

    staple11, staple12, staple13 = np.split(firststaplematricies, 3, axis=4)
    staple21, staple22, staple23 = np.split(secondstaplematricies, 3, axis=4)


    # calculate the twisted staple for each nu (index 2)


    firststaple = Barray[..., None, None][start:end+1] * staple11 @ staple12.conj().swapaxes(-1,-2) @ staple13.conj().swapaxes(-1, -2)
    secondstaple = V2_Barray[..., None, None][start:end+1] * staple21.conj().swapaxes(-1,
                                                                                          -2) @ staple22.conj().swapaxes(
        -1, -2) @ staple23


    # adding two terms together to get full staple then summing to get Vmu array. Squeeze to get rid of vestigal indicies
    staplesum = firststaple + secondstaple

    Varray = np.sum(staplesum, axis=2)
    Varray = np.squeeze(Varray)


    # calculating momentum update
    stapleterm = link_array @ Varray
    momentum_change = (1 / g ** 2) * (stapleterm.conj().swapaxes(-1, -2) - stapleterm) * dt

    new_momentum_array = momentum_array + momentum_change

    new_momentum_array = np.squeeze(new_momentum_array)

    return new_momentum_array


def batched_single_node_link_update(dt, start, end):
        global shared_config

        config = shared_config

        link_array = config[0]
        momentum_array = config[1]

        batch_link = link_array[start:end+1]
        batch_momentum = momentum_array[start:end+1]

        batch_out = su2_exp(dt * batch_momentum) @ batch_link
        return batch_out


def parallel_evolution_step(_pool, config, staple_matrix_array, staple_index_array, dt, batch_size, num_nodes, processes, g):
    global shared_config
    global shared_staple_matricies

    pool = _pool
    link_array = shared_config[0]

    array_shape = np.shape(link_array)

    new_momentum_array = np.empty(array_shape, dtype=link_array.dtype)
    new_link_array = np.empty(array_shape, dtype=link_array.dtype)


    start = time.time()
    momentum_inputs = [[dt, i * batch_size, np.min([(i + 1) * batch_size-1, num_nodes-1]), g] for i in range(processes)]
    #new_momentum_array = np.stack(pool.starmap(batched_single_node_momentum_update, momentum_inputs))
    for i, result in enumerate(pool.starmap(batched_single_node_momentum_update, momentum_inputs)):
        new_momentum_array[i * batch_size:(i + 1) * batch_size] = result
    print("momentum updated in", time.time()-start)

    #new_momentum_array = np.reshape(new_momentum_array, newshape = array_shape)
    shared_config[1] = new_momentum_array
    #print("momentum updated in", time.time()-start)

    start = time.time()
    link_inputs = [[dt,i * batch_size, np.min([(i + 1) * batch_size-1, num_nodes-1])] for i in range(processes)]
    #new_link_array = np.stack(pool.starmap(batched_single_node_link_update, link_inputs))
    for i, result in enumerate(pool.starmap(batched_single_node_link_update, link_inputs)):
        new_link_array[i * batch_size:(i + 1) * batch_size] = result

    #new_link_array = np.reshape(new_link_array, newshape=array_shape)
    #print("link updated in", time.time()-start)


    start = time.time()
    shared_config[0] =new_link_array
    #print("stacked!", time.time()-start)

    start = time.time()
    shared_staple_matricies = _make_staple_array(staple_index_array, shared_config[0])
    #print("staple matrix array updated", time.time()-start)














def _parallel_time_evolve(initial_config, dt, staple_index_array, input_Barray, input_V2_Barray, g, processes, nsteps=10000):
    global shared_config
    global shared_staple_matricies
    global shared_staple_index
    global Barray
    global V2_Barray

    Barray= input_Barray
    V2_Barray = input_V2_Barray
    link_array = initial_config[0]

    shared_staple_index = staple_index_array
    num_nodes = np.shape(Barray)[0]

    batch_size = int(num_nodes/processes) #number of nodes over number of processes, should be an integer for ease of life


    array_shape = np.shape(initial_config[0])


    shared_staple_matricies = _make_staple_array(shared_staple_index, link_array)


    with mp.Pool(processes = processes) as pool:
        _pool = pool



        shared_config = np.asarray([link_array.copy(), initial_config[1].copy()])
        momen_start = time.time()
        momentum_inputs = [[dt/2, i * batch_size, np.min([(i + 1) * batch_size - 1, num_nodes - 1]), g] for i in
                           range(processes)]
        new_momentum_array = np.stack(pool.starmap(batched_single_node_momentum_update, momentum_inputs))
        new_momentum_array = np.reshape(new_momentum_array, newshape=array_shape)
        shared_config = np.stack([link_array, new_momentum_array])

        link_start = time.time()
        link_inputs = [[dt, i * batch_size, np.min([(i + 1) * batch_size - 1, num_nodes - 1])] for i in
                       range(processes)]

        new_link_array = np.stack(pool.starmap(batched_single_node_link_update, link_inputs))
        new_link_array = np.reshape(new_link_array, newshape=array_shape)
        shared_config = np.stack([new_link_array, new_momentum_array])
        shared_staple_matricies = _make_staple_array(staple_index_array, shared_config[0])

        for i in range(nsteps - 1):
            #print("step", i)
            parallel_evolution_step(_pool, shared_config, shared_staple_matricies, shared_staple_index, dt, batch_size, num_nodes, processes, g)

        momentum_inputs = [[dt / 2, i * batch_size, np.min([(i + 1) * batch_size - 1, num_nodes - 1]), g] for i in
                           range(processes)]
        new_momentum_array = np.stack(pool.starmap(batched_single_node_momentum_update, momentum_inputs))
        new_momentum_array = np.reshape(new_momentum_array, newshape=array_shape)

        shared_config = np.stack([shared_config[0], new_momentum_array])

    output_config = shared_config.copy()


    return output_config



