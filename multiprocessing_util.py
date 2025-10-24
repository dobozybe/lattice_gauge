import os

import utilities

"""os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
"""
from utilities import *
import multiprocessing as mp
import multiprocessing.shared_memory as shared_memory
import numpy as np
import time

this_momen_time = []
this_link_time = []



def _make_staple_array(staple_index_array, link_array):
    staple_matrix_array = np.zeros(np.shape(staple_index_array) + (2,), dtype=np.complex128)

    # Create a mask for where index[1] != index[2] (non-diagonal elements)
    mask = np.ones(staple_index_array.shape[:3], dtype=np.bool_)
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


def batched_single_node_momentum_update(dt, start,end, g, action): #ND numpy array nodecoords


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


    new_momentum_array = momentum_array + momentum_change * (1 + extra_action_term_derivative(action))

    new_momentum_array = np.squeeze(new_momentum_array)

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






def batched_single_node_link_update(dt, start, end):
        global config

        link_array = config[0]
        momentum_array = config[1]

        batch_link = link_array[start:end+1]
        batch_momentum = momentum_array[start:end+1]

        batch_out = su2_exp(dt * batch_momentum) @ batch_link
        return batch_out


def parallel_evolution_step(_pool, config, staple_matrix_array, staple_index_array, dt, batch_size, num_nodes, processes, g, Barray):
    #print('starting evolution step')
    pool = _pool
    link_array = config[0]

    array_shape = np.shape(link_array)

    action = (1/g**2) * get_wilson_action(configuration=config, staple_index=staple_index_array, barray=Barray)
    start = time.time()
    momentum_inputs = [[dt, i * batch_size, np.min([(i + 1) * batch_size-1, num_nodes-1]), g, action] for i in range(processes)]
    new_momentum_array = np.stack(pool.starmap(batched_single_node_momentum_update, momentum_inputs))
    new_momentum_array = np.reshape(new_momentum_array, newshape = array_shape)
    config[:] = np.stack([link_array, new_momentum_array])
    this_momen_time.append(time.time()-start)

    start = time.time()
    link_inputs = [[dt,i * batch_size, np.min([(i + 1) * batch_size-1, num_nodes-1])] for i in range(processes)]
    new_link_array = np.stack(pool.starmap(batched_single_node_link_update, link_inputs))
    new_link_array = np.reshape(new_link_array, newshape=array_shape)
    this_link_time.append(time.time()-start)


    start = time.time()
    config[:] = np.stack([new_link_array, new_momentum_array])
    #print("stacked!", time.time()-start)

    start = time.time()
    staple_matrix_array[:] = _make_staple_array(staple_index_array, config[0])[:]
    #print("staple matrix array updated", time.time()-start)












def _worker_init(shm_staple_matrix_array_name, shm_config_name, shm_Barray_name, shm_V2_Barray_name, shm_staple_index_array_name, staple_matrix_shape, staple_index_shape, config_shape, Barray_shape, config_dtype, Barray_dtype, staple_index_dtype):
    global config
    global Barray
    global V2_Barray
    global staple_index_array
    global staple_matrix_array

    global config_mem
    global Barray_mem
    global V2_Barray_mem
    global staple_index_mem
    global staple_matrix_mem


    config_mem = shared_memory.SharedMemory(name = shm_config_name)
    Barray_mem = shared_memory.SharedMemory(name=shm_Barray_name)
    V2_Barray_mem = shared_memory.SharedMemory(name=shm_V2_Barray_name)
    staple_index_mem = shared_memory.SharedMemory(name=shm_staple_index_array_name)
    staple_matrix_mem=shared_memory.SharedMemory(name=shm_staple_matrix_array_name)

    config = np.ndarray(config_shape, dtype = config_dtype, buffer = config_mem.buf)
    Barray = np.ndarray(Barray_shape, dtype=Barray_dtype, buffer=Barray_mem.buf)
    V2_Barray = np.ndarray(Barray_shape, dtype=Barray_dtype, buffer=V2_Barray_mem.buf)
    staple_index_array = np.ndarray(staple_index_shape, dtype=staple_index_dtype, buffer = staple_index_mem.buf)
    staple_matrix_array = np.ndarray(staple_matrix_shape, dtype=np.complex128, buffer=staple_matrix_mem.buf)
    #print("process initialized!!")

    #return config_mem, Barray_mem, V2_Barray_mem, staple_index_mem, staple_matrix_mem, config, Barray, V2_Barray, staple_index_array, staple_matrix_array


def _parallel_time_evolve(initial_config, dt, staple_index_array, Barray, V2_Barray, g, processes, nsteps=10000):

    num_nodes = np.shape(Barray)[0]

    batch_size = int(num_nodes/processes) #number of nodes over number of processes


    shm_config = shared_memory.SharedMemory(create=True, size=initial_config.nbytes)
    shared_config = np.ndarray(initial_config.shape, dtype=initial_config.dtype, buffer=shm_config.buf)
    shared_config[:] = initial_config[:]

    shm_staple = shared_memory.SharedMemory(create=True, size=staple_index_array.nbytes)
    shared_staple_index = np.ndarray(staple_index_array.shape, dtype=staple_index_array.dtype, buffer=shm_staple.buf)
    shared_staple_index[:] = staple_index_array[:]

    shm_Barray = shared_memory.SharedMemory(create=True, size=Barray.nbytes)
    shared_Barray = np.ndarray(Barray.shape, dtype=Barray.dtype, buffer=shm_Barray.buf)
    shared_Barray[:] = Barray[:]

    shm_V2_Barray = shared_memory.SharedMemory(create=True, size=V2_Barray.nbytes)
    shared_V2_Barray = np.ndarray(V2_Barray.shape, dtype=V2_Barray.dtype, buffer=shm_V2_Barray.buf)
    shared_V2_Barray[:] = V2_Barray[:]


    staple_matrix_array = _make_staple_array(staple_index_array, initial_config[0])
    shm_staple_matricies = shared_memory.SharedMemory(create=True, size=staple_matrix_array.nbytes)
    shared_staple_matricies = np.ndarray(np.shape(staple_index_array) + (2,), dtype=np.complex128, buffer=shm_staple_matricies.buf)
    shared_staple_matricies[:] = staple_matrix_array[:]

    array_shape = np.shape(initial_config[0])

    with mp.Pool(processes = processes,
                 initializer= _worker_init,
                 initargs=(shm_staple_matricies.name, shm_config.name, shm_Barray.name, shm_V2_Barray.name,
                           shm_staple.name, np.shape(staple_matrix_array), np.shape(staple_index_array), np.shape(initial_config),
                           np.shape(Barray), initial_config.dtype, Barray.dtype, staple_index_array.dtype)) as pool:
        _pool = pool


        link_array = shared_config[0]


        #print("first momentum update")
        momen_start = time.time()
        action = (1/g**2) * get_wilson_action(configuration=initial_config, staple_index=staple_index_array, barray = Barray)
        momentum_inputs = [[dt/2, i * batch_size, np.min([(i + 1) * batch_size - 1, num_nodes - 1]), g, action] for i in
                           range(processes)]
        new_momentum_array = np.stack(pool.starmap(batched_single_node_momentum_update, momentum_inputs))
        new_momentum_array = np.reshape(new_momentum_array, newshape=array_shape)
        shared_config[:] = np.stack([link_array, new_momentum_array])
        #print("end of first momentum update")



        link_start = time.time()
        link_inputs = [[dt, i * batch_size, np.min([(i + 1) * batch_size - 1, num_nodes - 1])] for i in
                       range(processes)]

        new_link_array = np.stack(pool.starmap(batched_single_node_link_update, link_inputs))
        new_link_array = np.reshape(new_link_array, newshape=array_shape)
        shared_config[:] = np.stack([new_link_array, new_momentum_array])
        shared_staple_matricies[:] = _make_staple_array(staple_index_array, shared_config[0])[:]

        action2 = (1 / g ** 2) * get_wilson_action(configuration=shared_config, staple_index=staple_index_array, barray=Barray)
        #print("action change after first two steps", action2-action)
        evstep = time.time()
        for i in range(nsteps - 1):
            #print("step", i)
            parallel_evolution_step(_pool, shared_config, shared_staple_matricies, shared_staple_index, dt, batch_size, num_nodes, processes, g, Barray)
        print("time for evolution steps", time.time()-evstep)
        print("average momentum time", np.average(this_momen_time))
        print("average link time", np.average(this_link_time))
        #print("last action in time evolve")
        action = (1/g**2) * get_wilson_action(configuration=shared_config, staple_index=staple_index_array, barray = Barray)


        momentum_inputs = [[dt / 2, i * batch_size, np.min([(i + 1) * batch_size - 1, num_nodes - 1]), g, action] for i in
                           range(processes)]
        new_momentum_array = np.stack(pool.starmap(batched_single_node_momentum_update, momentum_inputs))
        new_momentum_array = np.reshape(new_momentum_array, newshape=array_shape)
        shared_config[:] = np.stack([link_array, new_momentum_array])

    output_config = shared_config.copy()
    shm_config.close()
    shm_config.unlink()
    shm_Barray.close()
    shm_Barray.unlink()
    shm_V2_Barray.close()
    shm_V2_Barray.unlink()
    shm_staple.close()
    shm_staple.unlink()
    shm_staple_matricies.close()
    shm_staple_matricies.unlink()

    return output_config



