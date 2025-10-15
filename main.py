from lattice_mp import *
from lattice import randomSU2
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
import cProfile
import pickle
import tracemalloc
import time
from observables import *
import multiprocessing
import os




def dicts_equal(d1, d2, tol=True):
    if d1.keys() != d2.keys():
        print("Different keys:")
        print("Only in d1:", d1.keys() - d2.keys())
        print("Only in d2:", d2.keys() - d1.keys())
        return False
    cmp = np.allclose if tol else np.array_equal
    equal = True
    for k in d1:
        if not cmp(d1[k], d2[k]):
            print(f"Mismatch at key: {k}")
            print("d1 value:", d1[k])
            print("d2 value:", d2[k])
            equal = False
    return equal

#random momentum, on average, gives an energy of 124k. Meaning we can expect action to jump that much during time evolution.
# Thus, for a large lattice, we need 124k/S_BPS < 0.01 for the simulation to stay within 1% of the BPS limit.
#S_BPS = 4 pi^2/g^2, so g^2 * 3140 < 0.01, or g ~ 0.00178457557
if __name__ == "__main__":

    twistmatrix = [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]]

    if not np.all((np.array(twistmatrix) + np.array(twistmatrix).T) == 0):
        print("Bad twist matrix!")
        sys.exit()

    myLattice = Lattice([24,6,6,24], twistmatrix = twistmatrix) #24,6,6,24
    myLattice.processes = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    myLattice.chunksize = 6*6*6*12
    plaquette = One_Cube_Plaquette()
    windingGeneral = General_Winding([0,0,0,0])
    action = Action()
    topcharge = TopologicalCharge()
    #myLattice.g = 0.0017
    myLattice.g=1
    print("expected BPS", (4 * np.pi**2/myLattice.g**2))

    #myLattice.reduce_action(2, 1, 1000, log = False)
    #myLattice.save_links("low_action_lattice_1")
    myLattice.parallel_chain(1, 1, 1000, observables=[action, windingGeneral, topcharge])
    #myLattice.chain(1,1,100, log = False)



    #data = handle_observables("14-10-2025_10:13:08['genwinding[0, 0, 0, 0]', 'topcharge', 'action'][24, 6, 6, 24]_twists:[[0, 3], [1, 2]]", [action, windingGeneral, topcharge])
    #action.visualize(data)
    #windingGeneral.visualize(data)
    #topcharge.visualize(data)
    #plt.show()





"""momentum = myLattice.random_momentum()
    holder = []
    for arraylist in list(momentum.values()):
        newlist = []
        for array in arraylist:
            newlist.append(array.copy())
        holder.append(newlist)
    old_config = [myLattice.get_link_matrix_dict(), dict(zip(momentum.keys(), holder))]

    config = [myLattice.get_link_matrix_dict(), momentum]
    new_config1 = myLattice.momentum_update(config, 0.01)

    link_dict = old_config[0]
    momentum_dict = old_config[1]

    link_array = np.array(list(link_dict.values()))
    momentum_array = np.array(list(momentum_dict.values()))

    staple_matricies = np.array([link_array[link.node1.index_in_real_nodearray][
                                     link.direction] if link != None else np.array([[0, 0], [0, 0]]) for link in
                                 myLattice.staple_array.flatten()]).reshape(myLattice.staple_array.shape + (2, 2))


    new_config2 = myLattice.batched_single_node_momentum_update([link_array, momentum_array], 0.01, 0, len(myLattice.real_nodearray)-1, staple_matricies)


"""



"""config = [myLattice.get_link_matrix_dict(), myLattice.random_momentum()]



    momentum = myLattice.vectorized_momentum_update(config, 0.1)[1]

    holder = []
    for arraylist in list(momentum.values()):
        newlist = []
        for array in arraylist:
            newlist.append(array.copy())
        holder.append(newlist)

    momentum1 = dict(zip(momentum.keys(), holder))

    momentum2 = myLattice.parallel_momentum_update(config, 0.1)[1]


    print(dicts_equal(momentum1, momentum2))
"""


def create_saved_lattice():
    myLattice = Lattice([3,3,3,3])
    momentum = myLattice.random_momentum()
    myLattice.save_links("saved_test_lattice")
    lattice_config = list(momentum.values())
    with open("savedmomentum.pkl", "wb") as f:
        pickle.dump(lattice_config, f)
    return

def testchain(number_iterations, evolution_time, number_steps):
    starttime = time.time()
    with open("savedmomentum.pkl", "rb") as f:
        momentumvals = pickle.load(f)
    myLattice = Lattice([3,3,3,3], filename="saved_test_lattice")
    acceptances = 0
    momentum = dict(zip(myLattice.get_link_matrix_dict().keys(),momentumvals))
    for i in range(number_iterations):
        holder = []
        for arraylist in list(momentum.values()):
            newlist = []
            for array in arraylist:
                newlist.append(array.copy())
            holder.append(newlist)
        old_config = [myLattice.get_link_matrix_dict(), momentum]
        candidate = myLattice.generate_candidate_configuration(old_config, evolution_time, number_steps)
        old_config = [myLattice.get_link_matrix_dict(), dict(zip(momentum.keys(), holder))]
        if myLattice.accept_config(candidate, old_config):
            print("accepted")
            acceptances += 1
        momentum = myLattice.random_momentum()
    print("Avg time per config: ", (time.time() - starttime) / number_iterations)
    print("acceptance rate:", acceptances / number_iterations)
    return candidate

def cool_measure(iterations, observables, log =True, filename = None):
    observable_list = []
    lattice_shape = [4,4,4,4]
    for i in range(iterations):
        observable_dict = {}
        thisLattice = Lattice(lattice_shape, twistmatrix=twistmatrix)
        thisLattice.action_min_sweep(1000)
        for observable in observables:
            observable_dict[observable.identifier] = tuple(observable.evaluate(thisLattice))
        observable_list.append(observable_dict)
    if (log == True or filename != None):
        if filename != None:
            with open("Cooling " + filename + str([obs.identifier for obs in observables]) + str(lattice_shape) + ".txt", "w") as f:
                for item in observable_list:
                    f.write(f"{item}\n")
        else:
            with open("Cooling " + str(time.strftime("%d-%m-%Y_%H:%M:%S")) + str([obs.identifier for obs in observables]) + str(
                    lattice_shape) + ".txt", "w") as f:
                for item in observable_list:
                    f.write(f"{item}\n")



