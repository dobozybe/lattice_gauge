from lattice import Lattice
from lattice import randomSU2
import matplotlib.pyplot as plt
import numpy as np
import copy


myLattice = Lattice([7,7,7,7])

myLattice.action_min_sweep(70)


