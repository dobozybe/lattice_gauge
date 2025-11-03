from lattice_mp import *
from lattice import randomSU2
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
import cProfile
import pickle
import tracemalloc
import time
from observables import *
import multiprocessing
import os
os.environ['PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;' + os.environ['PATH']
#from numba import cuda
from utilities import *





def dicts_equal(d1, d2, tol=True):
    if d1.keys() != d2.keys():
        print("Different keys:")
        print("Only in d1:", d1.keys() - d2.keys())
        print("Only in d2:", d2.keys() - d1.keys())
        return False
    cmp = np.allclose if tol else np.array_equal
    equal = True
    for k in d1:
        if not cmp(d1[k], d2[k]):
            print(f"Mismatch at key: {k}")
            print("d1 value:", d1[k])
            print("d2 value:", d2[k])
            equal = False
    return equal

#random momentum, on average, gives an energy of 124k. Meaning we can expect action to jump that much during time evolution.
# Thus, for a large lattice, we need 124k/S_BPS < 0.01 for the simulation to stay within 1% of the BPS limit.
#S_BPS = 4 pi^2/g^2, so g^2 * 3140 < 0.01, or g ~ 0.00178457557
if __name__ == "__main__":


    twistmatrix = [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]]

    if not np.all((np.array(twistmatrix) + np.array(twistmatrix).T) == 0):
        print("Bad twist matrix!")
        sys.exit()

    myLattice = Lattice([8,4,4,8], twistmatrix = twistmatrix, filename = "low_action_8448") #24,6,6,24
    #myLattice= Lattice([8,4,4,8], twistmatrix=twistmatrix, filename="low_action_8448")
    myLattice.processes = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    windingGeneral = General_Winding([0,0,0,0])
    action = Action()
    topcharge = TopologicalCharge()
    myLattice.g = 1
    #myLattice.g = 0.01
    print("Expected BPS:", 4 * np.pi**2/myLattice.g**2)
    #myLattice.reduce_action(2, 1, 100)

    print(myLattice.parallel_chain(4, 1, 1000, observables=[action, windingGeneral, topcharge]))
    #myLattice.chain(1, 1, 1000, observables=[action, windingGeneral, topcharge])




    #data = handle_observables("14-10-2025_10:13:08['genwinding[0, 0, 0, 0]', 'topcharge', 'action'][24, 6, 6, 24]_twists:[[0, 3], [1, 2]]", [action, windingGeneral, topcharge])
    #action.visualize(data)
    #windingGeneral.visualize(data)
    #topcharge.visualize(data)
    #plt.show()


