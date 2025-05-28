from lattice import Lattice
from lattice import randomSU2
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys


twistmatrix = [[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]]
"""
myLattice = Lattice([24,6,6,24], twistmatrix = twistmatrix)
myLattice.action_min_sweep(1000)
myLattice.save_links("twistedmin")
"""

myLattice = Lattice([24,6,6,24],twistmatrix=twistmatrix, filename = "twistedmin") #expect action of 39.48
myLattice.action_density_plot([0,3],[1,1])
#myLattice.action_min_sweep(3500) #total runs: 10000
#myLattice.save_links("twistedmin")

"""
myLattice.action_density_plot([0,3],[1,1])
#print("Calculated Action", myLattice.get_action())
"""
"""
myLattice = Lattice([10,10,10])
"""