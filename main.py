from lattice import Lattice
from lattice import randomSU2
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
import shelve

twistmatrix = [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]
myLattice = Lattice([24,6,6,24], twistmatrix = twistmatrix)

myLattice.action_min_sweep(1000)
myLattice.save_links("savedlattice.db")






#myLattice.action_min_sweep(1000)
#print("Calculated Action", myLattice.get_action())


