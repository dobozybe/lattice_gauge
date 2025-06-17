from lattice import Lattice
from lattice import randomSU2
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
import cProfile




twistmatrix = [[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]]

myLattice = Lattice([24,6,6,24], twistmatrix = twistmatrix, filename="twistedmin")

myLattice.action_density_plot([0,3],[2,2])

"""

myLattice.action_min_sweep(10000) #total runs: 20000
myLattice.save_links("twistedmin")"""
