from lattice import Lattice
from lattice import randomSU2
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
import cProfile
import time




twistmatrix = [[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]]

myLattice = Lattice([24,6,6,24], twistmatrix = twistmatrix)

starttime = time.time()
y=myLattice.get_action()
print(y)
print("normal action time:", time.time()-starttime)

"""for plane in myLattice.planeslist:
    if plane == [0,1]:
        print("plane:", plane, " ", myLattice.get_plaquette_holonomy(myLattice.real_nodearray[12].coordinates, plane))

"""
starttime = time.time()
x=myLattice.hamiltonian(myLattice.link_dict)
print(x)
print("hamiltonian action time:", time.time()-starttime)


print("per-node difference: ", np.abs(x-y)/len(myLattice.real_nodearray))
"""

myLattice.action_min_sweep(10000) #total runs: 20000
myLattice.save_links("twistedmin")"""
