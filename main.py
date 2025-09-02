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

##TODO: Increase acceptance rate
##TODO: Clean up Momentum Update Code
##TODO: Better Hamiltonian Calculation for graph

twistmatrix = [[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]]

#myLattice = Lattice([3,3,3,3], twistmatrix = twistmatrix) #24,6,6,24



def create_saved_lattice():
    myLattice = Lattice([3,3,3,3])
    momentum = myLattice.random_momentum()
    myLattice.save_links("saved_test_lattice")
    lattice_config = list(momentum.values())
    with open("savedmomentum.pkl", "wb") as f:
        pickle.dump(lattice_config, f)
    return



#x = myLattice.chain(1, 5, 50)






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



#create_saved_lattice()
#testchain(1, 10, 100) #seem to get a ~60% acceptance rate with a dt of 0.003?!

myLattice = Lattice([2,2,2,2], twistmatrix = twistmatrix)
momentum = myLattice.random_momentum()
config = [myLattice.get_link_matrix_dict(), momentum]
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
"""y, Vy = myLattice.momentum_update(config, 0.01)
x, Vx = myLattice.vectorized_momentum_update(config, 0.01)
y = y[1]
x = x[1]
print(dicts_equal(x,y))

"""
#print(dicts_equal(x,y))

myLattice.chain(5, 10,400)
plt.show()