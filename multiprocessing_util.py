import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np



import numpy as np
from utilities import *
import multiprocessing as mp
import multiprocessing.shared_memory as shared_memory



def _make_staple_array(staple_index_array, link_array):
    staple_matrix_array = np.zeros(np.shape(staple_index_array) + (2,), dtype=np.complex128)
    for index, val in np.ndenumerate(staple_index_array):
        staple_index = index[:-1]
        if index[1] != index[2]:
            entry =  staple_index_array[staple_index]
            staple_matrix_array[staple_index] = link_array[tuple(entry)]
        else:
            continue
    return staple_matrix_array


def batched_single_node_momentum_update(dt, start,end, g): #ND numpy array nodecoords


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
        global config

        link_array = config[0]
        momentum_array = config[1]

        batch_link = link_array[start:end+1]
        batch_momentum = momentum_array[start:end+1]

        batch_out = su2_exp(dt * batch_momentum) @ batch_link
        return batch_out


def parallel_evolution_step(_pool, config, staple_matrix_array, staple_index_array, dt, batch_size, num_nodes, processes, g):
    pool = _pool
    link_array = config[0]

    array_shape = np.shape(link_array)

    momentum_inputs = [[dt, i * batch_size, np.min([(i + 1) * batch_size-1, num_nodes-1]), g] for i in range(processes)]
    new_momentum_array = np.stack(pool.starmap(batched_single_node_momentum_update, momentum_inputs))
    new_momentum_array = np.reshape(new_momentum_array, newshape = array_shape)
    config[:] = np.stack([link_array, new_momentum_array])

    link_inputs = [[dt,i * batch_size, np.min([(i + 1) * batch_size-1, num_nodes-1])] for i in range(processes)]
    new_link_array = np.stack(pool.starmap(batched_single_node_link_update, link_inputs))
    new_link_array = np.reshape(new_link_array, newshape=array_shape)
    config[:] = np.stack([new_link_array, new_momentum_array])
    staple_matrix_array[:] = _make_staple_array(staple_index_array, config[0])[:]












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

        momentum_inputs = [[dt/2, i * batch_size, np.min([(i + 1) * batch_size - 1, num_nodes - 1]), g] for i in
                           range(processes)]
        new_momentum_array = np.stack(pool.starmap(batched_single_node_momentum_update, momentum_inputs))
        new_momentum_array = np.reshape(new_momentum_array, newshape=array_shape)
        shared_config[:] = np.stack([link_array, new_momentum_array])



        link_inputs = [[dt, i * batch_size, np.min([(i + 1) * batch_size - 1, num_nodes - 1])] for i in
                       range(processes)]

        new_link_array = np.stack(pool.starmap(batched_single_node_link_update, link_inputs))
        new_link_array = np.reshape(new_link_array, newshape=array_shape)
        shared_config[:] = np.stack([new_link_array, new_momentum_array])
        shared_staple_matricies[:] = _make_staple_array(staple_index_array, shared_config[0])[:]


        for i in range(nsteps - 1):
            parallel_evolution_step(_pool, shared_config, shared_staple_matricies, shared_staple_index, dt, batch_size, num_nodes, processes, g)

        momentum_inputs = [[dt / 2, i * batch_size, np.min([(i + 1) * batch_size - 1, num_nodes - 1]), g] for i in
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



