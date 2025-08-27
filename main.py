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



twistmatrix = [[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]]

myLattice = Lattice([3,3,3,3], twistmatrix = twistmatrix) #24,6,6,24

#myLattice = Lattice([2,2])


"""for node in myLattice.real_nodearray:
    for link in node.links:
        thislink = link[0]
        for direction in range(myLattice.dimensions):
            thislink.set_matrix(np.array([[1,0],[0,1]], dtype = np.complex128))

print(myLattice.get_action())

config = [myLattice.get_link_matrix_dict(), myLattice.random_momentum()]

print(myLattice.get_config_action(config))"""

#x = myLattice.chain(1, 5, 50)




#Energy is conserved when: initial momentum is 0, or if we're on such a trajectory.
def Newchain(number_iterations, evolution_time, number_steps):
    starttime = time.time()
    with open("savedmomentum.pkl", "rb") as f:
        momentum = pickle.load(f)
    momentumvals = list(momentum.values())

    momentumvals = project(np.full(np.shape(np.array(list(momentum.values()))), 0j, dtype='complex128')) #action 175 momentum 85 issue issue
    # momentum = dict(zip(momentum.keys(),momentumvals))
    keys = list(myLattice.get_link_matrix_dict().keys())
    momentum = dict(zip(keys, momentumvals))
    acceptances = 0
    old_config = [myLattice.get_link_matrix_dict(), momentum]
    for i in range(number_iterations):
        candidate = myLattice.generate_candidate_configuration(old_config, evolution_time, number_steps)
        if myLattice.accept_config(candidate, old_config):
            print("accepted")
            acceptances += 1
        momentum = myLattice.random_momentum()
    print("Avg time per config: ", (time.time() - starttime) / number_iterations)
    print("acceptance rate:", acceptances / number_iterations)
    return candidate


x = myLattice.chain(1, 5, 800)
#y = Newchain(1, 5, 100)



plt.show()