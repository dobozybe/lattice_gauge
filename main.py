from lattice import Lattice
from lattice import randomSU2
import matplotlib.pyplot as plt
import numpy as np


#myLattice = Lattice([24,6,6,24])
myLattice = Lattice([10,10])


num_steps = 5
action_steps = myLattice.action_min_sweep(num_steps)


plt.scatter(range(num_steps+1),action_steps)
plt.show()



