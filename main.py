from lattice import Lattice
from lattice import randomSU2
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
import shelve

"""myLattice = Lattice([24,6,6,24])
myLattice.save_links("savedlattice")"""


myLattice = Lattice([24,6,6,24], "savedlattice")
print(myLattice[0,0,0,0].get_link(2,0).get_matrix())
print(myLattice[0,0,0,0].get_link(2,0).node1)




#myLattice.action_min_sweep(1000)
#print("Calculated Action", myLattice.get_action())


