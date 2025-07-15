import lattice
from lattice import *
from lattice import randomSU2
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
import cProfile
import time




twistmatrix = [[0,0,0,1],[0,0,1,0],[0,-1,0,0],[-1,0,0,0]]

myLattice = Lattice([24,6,6,24], twistmatrix = twistmatrix)

links = myLattice.get_link_matrix_dict()
momen = myLattice.random_momentum()
config = [links, momen]

"""def random_complex_matrix():
    real_part = np.random.randn(2, 2)
    imag_part = np.random.randn(2, 2)
    return real_part + 1j * imag_part

w = random_complex_matrix()
x = random_complex_matrix()
y = random_complex_matrix()
z = random_complex_matrix()

matrixarray = np.array([[x,y],[w,z]])

print(project(matrixarray))"""

myLattice.chain(5)
#myLattice.prealloc_momentum_update(config, 0.001)
#cProfile.runctx("myLattice.momentum_update(config, 0.001)", globals(), locals())
"""config = myLattice.time_evolve([myLattice.get_link_matrix_dict(), myLattice.random_momentum()], 3)

link_matrix_dict = config[0]
momentum_matrix_dict = config[1]

print(momentum_matrix_dict[myLattice[0,1,0,2]])"""

"""while True:
    newlinkdict = {}
    zeromomentumdict = {}
    for node in myLattice.real_nodearray:
        linkholder = []
        momentumholder = []
        for i in range(myLattice.dimensions):
            linkholder.append(randomSU2())
            momentumholder.append(np.array([[0,0],[0,0]]))
        newlinkdict[node] = linkholder
        zeromomentumdict[node]= momentumholder

    for node in myLattice.ghost_nodearray:
        linkholder = []
        momentumholder = []
        for i in range(myLattice.dimensions):
            if not node.ghost_node[i]:
                linkholder.append(newlinkdict[node.get_link(i, 0).parent_link.node1][i]) #I've already done the math about which links should be identified
                momentumholder.append(np.array([[0,0],[0,0]]))
            else:
                linkholder.append(np.array([[None,None],[None,None]])) #none array should never be involved in any calculation but should crash if it somehow is.
                momentumholder.append(np.array([[0, 0], [0, 0]]))
        newlinkdict[node] = linkholder
        zeromomentumdict[node] = momentumholder


    print(myLattice[0,0,0,0].get_link(1,0).get_matrix())
    print(myLattice[24,0,0,0].get_link(1,0).get_matrix())

    starttime = time.time()
    x = myLattice.accept_config([newlinkdict,zeromomentumdict],[myLattice.get_link_matrix_dict(), zeromomentumdict])
    print("elapsed:", time.time()-starttime)
    if x == True:
        print(myLattice[0,0,0,0].get_link(1,0).get_matrix())
        print(myLattice[24,0,0,0].get_link(1,0).get_matrix())
        break
    else:
        continue"""
