import lattice
from lattice import *
from lattice import randomSU2
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
import cProfile
import pickle
import time
from observables import *

##TODO: Better Hamiltonian Calculation for graph

twistmatrix = [[0,1,0,0],[-1,0,0,0],[0,0,0,1],[0,0,-1,0]]


if not np.all((np.array(twistmatrix) + np.array(twistmatrix).T) == 0):
    print("Bad twist matrix!")
    sys.exit()

myLattice = Lattice([1,1,1,1], twistmatrix = twistmatrix) #24,6,6,24


#observables for 1^4 lattice


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

import ast
def plot_HMC_out():
    with open("1cubesmalltime.txt", "r") as f:
        loaded_list = f.read().splitlines()
    truelist = []
    for entry in loaded_list:
        truelist.append(eval(entry, {"np": np}))
    onetwodata = []
    threefourdata = []
    for list in truelist:
        onetwodata.append(list[0][0][1])
        threefourdata.append(list[0][5][1])
    plt.figure()
    onetwocounts, onetwoedges = np.histogram(onetwodata, bins = 50)
    plt.bar(onetwoedges[:-1], onetwocounts, width = np.diff(onetwoedges))
    plt.title("Holonomy of the 1-2 plaquette")
    plt.figure()
    threefourcounts, threefouredges = np.histogram(threefourdata, bins = 50)
    plt.bar(threefouredges[:-1], threefourcounts, width = np.diff(threefouredges))
    plt.title("Holonomy of the 3-4 plaquette")
    plt.show()


#plot_HMC_out()
myLattice.chain(10000, 1, 400, observables=[plaquette_observable, winding_loop])


