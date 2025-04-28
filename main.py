from lattice import Lattice
from lattice import randomSU2
import matplotlib.pyplot as plt
import numpy as np


#myLattice = Lattice([24,6,6,24])
myLattice = Lattice([10,10,10,10])



num_steps = 30

action_steps, saved_action_steps, differences = myLattice.action_min_sweep(num_steps)


plt.scatter(range(num_steps+1),action_steps, label = "New method")
plt.scatter(range(num_steps+1),saved_action_steps, label = "Old Method")
plt.scatter(range(num_steps+1),differences, label = "Differences")
plt.legend()
plt.show()



