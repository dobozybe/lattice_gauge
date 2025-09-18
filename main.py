from lattice import *
from lattice import randomSU2
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
import cProfile
import pickle
import time
from observables import *




twistmatrix = [[0,1,0,0],[-1,0,0,0],[0,0,0,1],[0,0,-1,0]]



if not np.all((np.array(twistmatrix) + np.array(twistmatrix).T) == 0):
    print("Bad twist matrix!")
    sys.exit()

myLattice = Lattice([1,1,1,1], twistmatrix = twistmatrix) #24,6,6,24


plaquette = One_Cube_Plaquette()
windingGeneral = General_Winding([0,0,0,0])
action = Action()
topcharge = TopologicalCharge()



#myLattice.chain(1000000, 1, 100, observables=[windingGeneral, action, plaquette], filename = "million_1hypercube_0.7g")

#myLattice.cooling_measure(50, observables=[windingGeneral, action, plaquette, topcharge], filename="cooling_0.01")


data = handle_observables("million_1hypercube_0.7g['genwinding[0, 0, 0, 0]', 'action', 'onecubeplaquette'][1, 1, 1, 1]_twists:[[0, 1], [2, 3]]", [plaquette, windingGeneral, action])
plaquette.visualize(data)
action.visualize(data)
plt.show()

def create_saved_lattice():
    myLattice = Lattice([3,3,3,3])
    momentum = myLattice.random_momentum()
    myLattice.save_links("saved_test_lattice")
    lattice_config = list(momentum.values())
    with open("savedmomentum.pkl", "wb") as f:
        pickle.dump(lattice_config, f)
    return

def testchain(number_iterations, evolution_time, number_steps):
    starttime = time.time()
    with open("savedmomentum.pkl", "rb") as f:
        momentumvals = pickle.load(f)
    myLattice = Lattice([3,3,3,3], filename="saved_test_lattice")
    acceptances = 0
    momentum = dict(zip(myLattice.get_link_matrix_dict().keys(),momentumvals))
    for i in range(number_iterations):
        holder = []
        for arraylist in list(momentum.values()):
            newlist = []
            for array in arraylist:
                newlist.append(array.copy())
            holder.append(newlist)
        old_config = [myLattice.get_link_matrix_dict(), momentum]
        candidate = myLattice.generate_candidate_configuration(old_config, evolution_time, number_steps)
        old_config = [myLattice.get_link_matrix_dict(), dict(zip(momentum.keys(), holder))]
        if myLattice.accept_config(candidate, old_config):
            print("accepted")
            acceptances += 1
        momentum = myLattice.random_momentum()
    print("Avg time per config: ", (time.time() - starttime) / number_iterations)
    print("acceptance rate:", acceptances / number_iterations)
    return candidate

def cool_measure(iterations, observables, log =True, filename = None):
    observable_list = []
    lattice_shape = [4,4,4,4]
    for i in range(iterations):
        observable_dict = {}
        thisLattice = Lattice(lattice_shape, twistmatrix=twistmatrix)
        thisLattice.action_min_sweep(1000)
        for observable in observables:
            observable_dict[observable.identifier] = tuple(observable.evaluate(thisLattice))
        observable_list.append(observable_dict)
    if (log == True or filename != None):
        if filename != None:
            with open("Cooling " + filename + str([obs.identifier for obs in observables]) + str(lattice_shape) + ".txt", "w") as f:
                for item in observable_list:
                    f.write(f"{item}\n")
        else:
            with open("Cooling " + str(time.strftime("%d-%m-%Y_%H:%M:%S")) + str([obs.identifier for obs in observables]) + str(
                    lattice_shape) + ".txt", "w") as f:
                for item in observable_list:
                    f.write(f"{item}\n")



