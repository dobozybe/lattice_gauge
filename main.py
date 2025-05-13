from lattice import Lattice
from lattice import randomSU2
import matplotlib.pyplot as plt
import numpy as np
import copy


myLattice = Lattice([24,6,6,24])

myLattice.action_min_sweep(10)
print(myLattice.get_action())


