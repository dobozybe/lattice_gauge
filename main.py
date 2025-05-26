from lattice import Lattice
from lattice import randomSU2
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys

twistmatrix = [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]
myLattice = Lattice([24,6,6,24], twistmatrix = twistmatrix)
#myLattice.save_links("twistedmin.db")
myLattice.action_density_plot([0,1], [2,3])



#myLattice.action_min_sweep(1000)
#myLattice.save_links("twistedmin.db")





#myLattice.action_min_sweep(1000)
#print("Calculated Action", myLattice.get_action())


