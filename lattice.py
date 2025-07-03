import functools

import numpy as np
from node_links import *
import random
import matplotlib.pyplot as plt
import time
import shelve
import pickle
from sympy import LeviCivita
import scipy.stats
import scipy.linalg as linalg


def get_identity(position, direction):
    return np.diag(np.full(2, 1))

def randomSU2():
    alpha = random.random() * 2 * np.pi
    beta = random.random() * 2 * np.pi
    gamma = random.random() * 2 * np.pi
    a = np.exp(alpha*1j) * np.sin(beta)
    b = np.exp(gamma*1j) * np.cos(beta)
    matrix = np.array([[a, -np.conj(b)],[b, np.conj(a)]])
    return matrix


def inSU2(matrix):
    determinant1 = (np.round(np.linalg.det(matrix), 10) == np.float64(1))
    unitary = np.array_equal(np.round(matrix @ matrix.conj().T, 10),  np.diag(np.full(2, 1)))
    return determinant1 and unitary


class Lattice:  # ND torus lattice
    def __init__(self, shape, filename = None, twistmatrix = None):
        # dimensions should be an nd numpy array with each entry corresponding to the size of the array.
        # t should be the first dimension
        starttime = time.time()
        self.SUN_dimension = 2
        self.shape = shape
        self.filename = filename
        self.twistmatrix = np.zeros((len(shape),len(shape)))
        if twistmatrix!=None:
            self.twistmatrix = twistmatrix
        self.paddedshape = []
        self.planeslist = []

        self.indexdict ={}
        self.link_dict = {}

        self.num_nodes = 1 #not including ghosts
        for i in shape:
            self.num_nodes *= i


        for i in range(len(shape)): #make array bigger along each axis by one node. Outer padded layer is "ghost links"
            self.paddedshape.append(shape[i]+1)

        self.twistGenFunc = get_identity
        self.dimensions = len(shape)
        self.nodelist = []
        self.real_nodearray = [] #np array of nodes that aren't ghost nodes
        self.ghost_nodearray = []
        tuplelist = []
        self.make_tuples(np.zeros(len(shape)), self.paddedshape, tuplelist)


        for coordinates in tuplelist: #initialize nodes and add them to nodelist
            self.nodelist.append(Node(coordinates, is_ghost_node= [self.is_boundary(coordinates, i) for i in range(self.dimensions)]))#loops will ignore ghost nodes. If it's on the boundary it is a ghost node
            self.indexdict[tuple(coordinates)] = self.nodelist[self.get_node_index(coordinates)]

        for node in self.nodelist: #initialize real node array
            if not True in node.ghost_node:
                self.real_nodearray.append(node)
            else:
                self.ghost_nodearray.append(node)
        self.real_nodearray = np.array(self.real_nodearray)
        self.ghost_nodearray = np.array(self.ghost_nodearray)


        for i in range(self.dimensions): #initialize planeslist
            for j in range(i):
                self.planeslist.append([j,i])
        self.make_links()
        self.make_boundary_links()


        for node in self.nodelist:
            self.link_dict[node] = np.array(node.links).T[0]
        print("Initialized in:", time.time() - starttime)


    def get_node_index(self, coordinates):
        coords = coordinates[::-1]
        depths = self.paddedshape[::-1]
        runningsum = 0
        place_value = 1
        for i in range(len(depths[1:])):
            place_value *= depths[i]
            runningsum += int(coords[i + 1]) * place_value
        runningsum += coords[0]
        return int(runningsum)

    def __getitem__(self, coordinates):
        return self.indexdict[tuple(coordinates)]

    def set_twists(self, twistdict):
        self.twists.update(twistdict)
        return


    def translate(self, node, direction, distance): #given a node, return the node distance units later in a given direction
        position = node.coordinates.copy()
        position[direction] = (position[direction] + distance)%self.shape[direction]
        return self[position]

    def make_tuples(self, tuple, depths, outputlist):
        # generates a list of tuples in the nd array. Equivalent of "range" essentially. Only used to populate a dictionary during init
        if depths == []:
            outputlist.append(tuple.copy())
            return
        for j in range(depths[0]):
            tuple[-len(depths)] = j
            self.make_tuples(tuple, depths[1:], outputlist)

    def is_boundary(self, coordinates, direction): #todo: this feels inelegant
        if coordinates[direction] == self.shape[direction]:
            return True
        else:
            return False

    def make_links(self):
        if self.filename!= None:
            datafile = open(self.filename + ".pickle", "rb")
            matrixdict = pickle.load(datafile)
        for node in self.real_nodearray:
            # if we're not in the padded part of the array (not a ghost node) link to the next node in each direction
            for i in range(len(self.shape)):
                these_coords = node.coordinates
                new_coords = np.concatenate(
                    (these_coords[:i], [(these_coords[i] + 1)], these_coords[i + 1:]))
                newlink=Link(node, self[new_coords], i)
                if self.filename == None:
                    newlink.set_matrix(randomSU2())
                else:
                    this_matrix = matrixdict[str(these_coords) + ":" + str(i)]
                    newlink.set_matrix(this_matrix)
        return

    def make_student_link_adjacent(self, node, direction):
        next_coords = np.concat((node.coordinates[:direction], [node.coordinates[direction] + 1], node.coordinates[direction+1:]))
        new_link = StudentLink(node, self[next_coords], direction)
        return new_link

    def make_boundary_links(self): #todo: clean this up
        for i in range(self.dimensions):
            for node in self.nodelist:
                for j in range(self.dimensions):
                    if (node.ghost_node[i] and (not node.ghost_node[j])): #don't add links in the i direction (we're at the i boundary of the padded lattice)
                        new_link = self.make_student_link_adjacent(node, j)
                        identified_node_coordinates = node.coordinates.copy()
                        for k in range(self.dimensions):
                            identified_node_coordinates[k] = identified_node_coordinates[k] % self.shape[k]
                        identified_link = self[identified_node_coordinates].get_link(j, 0)
                        identified_link_matrix = identified_link.get_matrix()
                        #print(identified_link)
                        new_link.set_matrix(identified_link_matrix)
                        new_link.set_parent(identified_link)

    def B(self, mu, nu, node):
        position = node.coordinates
        if (position[mu] == self.shape[mu]-1) and (position[nu] == self.shape[nu]-1):
            return np.exp(-2 * np.pi * 1j * float(self.twistmatrix[mu][nu]) / self.SUN_dimension)
        else:
            return float(1)



    def save_links(self,filename):
        """while True:
            try:
                for node in self.nodelist:
                    for i in range(len(node.links)):
                        this_link = node.links[i][0]
                        with shelve.open(filename) as db:
                            #print(str(node.coordinates) + ":" + str(i))
                            db[str(node.coordinates) + ":" + str(i)] = this_link
                print("saved")
                break
            except:
                print("Broke at ", str(node.coordinates),".", "Trying again.")
                pass
        return 0"""
        matrixdict = {}
        for node in self.real_nodearray:
            for i in range(len(node.links)):
                matrix_id = str(node.coordinates) + ":" + str(i)
                matrixdict[matrix_id] = node.get_link(i, 0).get_matrix()
        with open(filename + ".pickle", "wb") as datafile:
            pickle.dump(matrixdict, datafile, protocol=pickle.HIGHEST_PROTOCOL)


    def get_link_matrix_dict(self): #returns only non-ghost link matricies
        matrix_dict = {}
        for node in self.real_nodearray:
            matrixholder = []
            for i in range(self.dimensions):
                thislink = node.get_link(i, 0)
                if thislink!= None:
                    matrixholder.append(thislink.get_matrix())
                else:
                    matrixholder.append(np.array([[None,None],[None,None]]))
            matrix_dict[node]=matrixholder
        return matrix_dict
    def get_plaquette_corners(self, node, plane): #with corner as bottom left point in plaquette
        cornercoords = node.coordinates
        first_corner = node
        second_corner = first_corner.get_next_node(plane[0], 0)
        third_corner = second_corner.get_next_node(plane[1], 0)
        fourth_corner = third_corner.get_next_node(plane[0], 1)
        return [first_corner, second_corner, third_corner, fourth_corner]

    def get_plaquette_links(self, node, plane): #ordered in correct orientation
        corners = self.get_plaquette_corners(node, plane)
        first_corner = corners[0]
        second_corner = corners[1]
        third_corner = corners[2]
        fourth_corner = corners[3]
        linklist = [
            first_corner.get_link(plane[0], 0),
            second_corner.get_link(plane[1], 0),
            third_corner.get_link(plane[0], 1),
            fourth_corner.get_link(plane[1], 1)
        ]
        return linklist

    def get_plaquette_matricies(self,node,plane):
        links = self.get_plaquette_links(node,plane)
        matrixlist = [
            links[0].get_matrix(),
            links[1].get_matrix(),
            links[2].get_matrix().conj().T,
            links[3].get_matrix().conj().T
        ]
        return matrixlist

    def get_plaquette_holonomy(self, cornercoords, plane):
        """plane should be 2 element list i.e. [1,3] if you want the plaquette in the 1,3 plane.
        We don't consider plaquettes that loop around the torus so don't consider these here. Therefore,
        the cornercoords shouldn't be edges of the plane you're in. This can be enforced by making sure they're
        smaller than the size of the lattice.
        """
        cornernode = self[cornercoords]
        product = np.diag([1,1])
        for matrix in self.get_plaquette_matricies(cornernode,plane):
            product = product @ matrix
        return product


    def get_action(self):
        action_sum = 0
        loop_count = 0
        num_plaquettes = len(self.planeslist) * self.num_nodes
        for node in self.real_nodearray:
            #print(node.coordinates)
            for plane in self.planeslist:
                loop_count +=1
                action_sum+= np.trace(self.B(plane[0],plane[1], node)*self.get_plaquette_holonomy(node.coordinates, plane)) #2.2 of FDW paper
        action = 2 * num_plaquettes - action_sum
        if np.imag(action) > 0.0001:
            print("Warning: Action is Complex")
            return action
        return np.real(action)

    def minimize_link_action(self,link):
        """Find all plaquettes the link contributes to"""
        Vdagsum = np.full((2,2), 0 + 0j) #for SU(2)
        #print("Start action:", start_action)
        for plane in self.planeslist:
            if link.direction in plane:
                this_plane = plane.copy()
                link_direction = link.direction
                this_plane.remove(link_direction)
                orthogonal_direction = this_plane[0]  # gives direction in plane orthogonal to link
                node = link.node1
                complement_node = self.translate(node, orthogonal_direction, -1) #other node whose plaquette in that plane will contain that link
                #This logic is good (checked thrice)
                Vdagmatrix_node = (self.B(plane[0],plane[1], node)* self.translate(node, link_direction, 1).get_link(orthogonal_direction, 0).get_matrix()
                                @ self.translate(node, orthogonal_direction, 1).get_link(link_direction, 0).get_matrix().conj().T
                                @ node.get_link(orthogonal_direction, 0).get_matrix().conj().T) #formula worked out in notebook (twice. This is verified).
                Vdagmatrix_complement = (np.conjugate(self.B(plane[0],plane[1], complement_node)) * self.translate(complement_node, link_direction, 1).get_link(orthogonal_direction, 0).get_matrix().conj().T
                                      @ complement_node.get_link(link_direction,0).get_matrix().conj().T
                                      @complement_node.get_link(orthogonal_direction, 0).get_matrix()
                                      )
                Vdagsum += Vdagmatrix_node
                Vdagsum += Vdagmatrix_complement
        Vdagsum[0][1] = -np.conj(Vdagsum[1][0]) #correcting entries of Vdagsum so that it doesn't drift from a real number times an SU2 matrix
        Vdagsum[1][1] = np.conj(Vdagsum[0][0])
        rsquared = np.real(np.linalg.det(Vdagsum))
        original_contrib = -np.trace(link.get_matrix() @ Vdagsum) #trace (U times vdag) is the original contribution to the action

        Wdag = Vdagsum/(np.sqrt(rsquared)) #since Wdag = Vsum/r, r = sqrt(det(Vsum)) (since det(Wdag = 1)). Then we can get Wdag from Vsum.
        new_contrib = - 2 * np.sqrt(rsquared) #should be equal to -2r
        #print("calculated:", new_contrib, " should be:", -2*np.sqrt(rsquared))
        """if not inSU2(Wdag):
            print("Warning! No longer in SU(2)!")
            return 1"""
        link.set_matrix(Wdag.conj().T) #set the matrix of our link to the dagger of Wdag. This should minimize this link's contribution to the action per FDW paper.
        #It should also deal with the reverse link matrix (since the link *is* the reverse link)
        if  link.student_links != []: #update the student if there is one
            for student_link in link.student_links:
                student_link.update()
        action_delta = new_contrib - original_contrib
        return np.real(action_delta)

    def lattice_sweep(self,initial_action):
        starttime = time.time()
        actionchanges = []
        action = initial_action
        original_action_changes = []
        contribution_counter = 0
        #print("Sweeping")
        for direction in range(len(self.shape)):
            for node in self.real_nodearray:
                action_change = self.minimize_link_action(node.get_link(direction, 0))
                actionchanges.append(action_change)
                action += action_change
        """
        for node in self.nodelist:
            if not (True in node.ghost_node):
                for link in node.getlinks():
                    action_change = self.minimize_link_action(
                        link[0])  # only use the forward facing link
                    actionchanges.append(action_change)
                    action += action_change  # this maybe will save time on needing to call get_action each loop?
        """
        print("Sweep Completed in", time.time()-starttime)
        return action

    def action_min_sweep(self, nsweeps): #1 sweep down to ~6 seconds. 1000 sweeps about 1.5 hours.
        start_time = time.time()
        actionlist = np.zeros(nsweeps + 1)
        start_action = self.get_action()
        actionlist[0] += start_action
        print("Starting Action:", actionlist[0])
        for i in range(nsweeps):
            print("Sweep ", i)
            new_action = self.lattice_sweep(actionlist[i])
            actionlist[i+1] += new_action
            print("Calculated action after sweep:", actionlist[i+1])
        plt.plot(actionlist, marker = "o", linestyle = "")
        print("total time:", time.time() - start_time)
        plt.show()
        return actionlist

    def action_density_plot(self, plane, planecoords): #plane coords should be other coordinates in order with the plane coordinates removed
        xlist= range(self.shape[plane[0]])
        ylist = range(self.shape[plane[1]])
        actionlist = np.zeros((self.shape[plane[1]], self.shape[plane[0]]))
        for node in self.real_nodearray:
            if (np.delete(np.delete(node.coordinates, plane[1]),plane[0]) == planecoords).all():
                this_location = [node.coordinates[plane[0]], node.coordinates[plane[1]]]
                this_action = 2 - np.trace(self.B(plane[0],plane[1], node)*self.get_plaquette_holonomy(node.coordinates, plane))
                actionlist[int(this_location[1])][int(this_location[0])] = np.real(this_action) #first index selects row and second selects column
        plt.pcolormesh(xlist, ylist, actionlist)
        titlestring = "with x"
        skipped = 0
        for i in range(self.dimensions):
            if i in plane:
                skipped +=1
                pass
            else:
                if i != self.dimensions - 1:
                    titlestring += str(i) + " = " + str(planecoords[i-skipped]) + ", x"
                else:
                    titlestring += str(i) + " = " + str(planecoords[i - skipped])
        plt.legend()
        plt.title("Action Density in the "+ str(plane)+ " plane, with " + titlestring)
        plt.show()

    def get_topological_charge(self):
        runningsum = 0
        indicies = [[i,j,k,l] for i in range(3) for j in range(3) for k in range(3) for l in range(3)]
        for node in self.real_nodearray:
                for index in indicies:
                    civita = LeviCivita(index[0], index[1],index[2],index[3])
                    if civita == 0:
                        continue
                    contribution = civita * np.trace(
                        self.get_plaquette_holonomy(node.coordinates, [index[0], index[1]])
                        @ self.get_plaquette_holonomy(node.coordinates, [index[2],index[3]])
                    )
                    runningsum += contribution
        return -1/(32 * np.pi**2) * runningsum


    """Hybrid Monte Carlo functions. 
    Will repeatedly use variable name "configuration" to represent a pair
     [link variable matrix dict, computer momentum dict]. It's important that the first entry
     is link variable matricies, and not the link variable objects themselves so we don't reset them
     during time evolution. Dictionaries must range over ***all*** nodes, including ghost. But code will
     only update non-ghost node values and will set ghost node values accordingly."""
    def ghost_fill_configuration(self,configuration):
        link_dict = configuration[0]
        momentum_dict = configuration[1]
        filled_link_dict = link_dict.copy()
        filled_momentum_dict = momentum_dict.copy()
        for node in self.ghost_nodearray:
            linkholder = []
            momentumholder = []
            for i in range(self.dimensions):
                if not node.ghost_node[i]:
                    linkholder.append(link_dict[node.get_link(i, 0).parent_link.node1][
                                          i])  # I've already done the math about which links should be identified
                    momentumholder.append(momentum_dict[node.get_link(i, 0).parent_link.node1][
                                          i])
                else:
                    linkholder.append(np.array([[None, None], [None,
                                                               None]]))  # none array should never be involved in any calculation but should crash if it somehow is.
                    momentumholder.append(np.array([[None, None], [None,
                                                               None]]))
            filled_link_dict[node] = linkholder
            filled_momentum_dict[node] = momentumholder
        new_config = [filled_link_dict, filled_momentum_dict]
        return new_config
    def random_momentum(self):
        new_momentum_dict = {}
        lie_gens =[
            np.array([[0,-1],[-1,0]]),
            np.array([[0,-1j],[1j,0]]),
            np.array([[-1,0],[0,-1]])
        ]
        for node in self.real_nodearray:
            momenta_list = []
            for direction in range(self.dimensions):
                momenta_vals = [np.random.normal(),np.random.normal(),np.random.normal()]
                momenta_list.append(momenta_vals[0] * lie_gens[0] + momenta_vals[1] * lie_gens[1]+momenta_vals[2] * lie_gens[2])
            new_momentum_dict[node]= momenta_list
        return new_momentum_dict
    def project(self, matrixarray):
        antisym=(matrixarray - np.transpose(matrixarray.conj(), axes = [0,2,1]))
        return (1/2) * antisym - (1/4)* np.trace(antisym) @ np.array([[1,0],[0,1]])

    def link_update(self, initial_config, dt): #sends link at n * dt to (n+1)*dt
        starttime = time.time()
        link_array = np.transpose(np.array(list(initial_config[0].values())), axes = [1,0,2,3])
        newlinks = []
        momentum_array = np.transpose(np.array(list(initial_config[1].values())), axes = [1,0,2,3])
        for direction in range(len(momentum_array)):
            link_array = scipy.linalg.expm(dt * momentum_array[direction]) @ link_array[direction]
            newlinks.append(link_array)
        newlinks = np.transpose(np.array(newlinks), axes = [1,0,2,3])
        new_link_dict = dict(zip(initial_config[0].keys(), list(newlinks)))
        new_config = [new_link_dict, initial_config[1]]
        print("link elapsed", time.time()-starttime)
        return new_config

    def momentum_update(self, initial_config, dt): #sends momentum at n*dt - dt/2 to n * dt + dt/2
        starttime = time.time()
        momentum_array = np.transpose(np.array(list(initial_config[1].values())), axes = [1,0,2,3])
        filled_config = self.ghost_fill_configuration(initial_config)
        filled_link_matricies_dict = filled_config[0]
        for direction in range(len(momentum_array)):
            plane_holonomies_sum = 0
            for plane in self.planeslist:
                if direction in plane:
                    node_plaquette_corners = []
                    Blist = []
                    starttime = time.time()
                    for node in self.real_nodearray:
                        node_plaquette_corners.append(self.get_plaquette_corners(node, plane))
                        Blist.append(self.B(plane[0], plane[1], node))
                    print("looptime", time.time()-starttime)
                    starttime = time.time()
                    node_plaquette_corners = np.array(node_plaquette_corners).T
                    Blist = np.array(Blist)
                    links0 = [filled_link_matricies_dict[node] for node in node_plaquette_corners[0]]
                    first_link_matricies = np.stack(
                        [link_matricies[plane[0]] for link_matricies in links0])
                    second_link_matricies = np.stack(
                        [link_matricies[plane[1]] for link_matricies in
                         [filled_link_matricies_dict[node] for node in node_plaquette_corners[1]]])
                    third_link_matricies = np.transpose(np.stack(
                        [link_matricies[plane[0]] for link_matricies in
                         [filled_link_matricies_dict[node] for node in node_plaquette_corners[3]]]), axes = [0,2,1])
                    fourth_link_matricies = np.transpose(np.stack(
                        [link_matricies[plane[1]] for link_matricies in links0]).conj(), axes = [0,2,1])
                    matricies = [first_link_matricies, second_link_matricies, third_link_matricies, fourth_link_matricies]
                    holonomies = functools.reduce(np.matmul, matricies)
                    plane_holonomies_sum += Blist[:, np.newaxis, np.newaxis] * holonomies
                    print("holonomyytime", time.time()-starttime)
            momentum_array[direction] -= 1/2 * dt * self.project(plane_holonomies_sum)
        new_momentum_dict = dict(zip(initial_config[0].keys(), list(np.transpose(np.array(momentum_array), axes = [1,0,2,3]))))
        new_config = [initial_config[0], new_momentum_dict]
        print("elapsed", time.time()-starttime)
        return new_config
    def evolution_step(self, config, dt):
        momentum_config = self.momentum_update(config, dt)
        link_config = self.link_update(momentum_config, dt)
        return link_config

    def time_evolve(self, initial_config, evolution_time,nsteps = 1000):
        starttime = time.time()
        config = initial_config
        dt = evolution_time/nsteps
        for i in range(nsteps):
            print(i)
            config = self.evolution_step(config, dt)
        print("Elapsed:", time.time()-starttime)
        return config


    def generate_candidate_configuration(self, current_configuration, evol_time):
        new_configuration = self.time_evolve(current_configuration, evol_time)
        return new_configuration

    def hamiltonian(self, configuration): #only takes a ghost filled configuration
        starttime = time.time()
        link_matricies_dict = configuration[0]
        momentum_dict = configuration[1]
        momentum_array = np.array(list(momentum_dict.values())).reshape(len(self.nodelist)*self.dimensions, self.SUN_dimension, self.SUN_dimension)

        #calculating action of configuration
        node_action_contribs = 0
        for plane in self.planeslist: #generate all plaquette holonomies
            node_plaquette_corners = []
            Blist = []
            for node in self.real_nodearray:
                node_plaquette_corners.append(self.get_plaquette_corners(node,plane))
                Blist.append(self.B(plane[0],plane[1], node))
            node_plaquette_corners = np.array(node_plaquette_corners).T
            Blist = np.array(Blist)
            first_link_matricies = np.array(
                [link_matricies[plane[0]] for link_matricies in [link_matricies_dict[node] for node in node_plaquette_corners[0]]])
            second_link_matricies = np.array(
                [link_matricies[plane[1]] for link_matricies in [link_matricies_dict[node] for node in node_plaquette_corners[1]]])
            third_link_matricies = np.array(
                [link_matricies[plane[0]].conj().T for link_matricies in [link_matricies_dict[node] for node in node_plaquette_corners[3]]])
            fourth_link_matricies = np.array(
                [link_matricies[plane[1]].conj().T for link_matricies in
                 [link_matricies_dict[node] for node in node_plaquette_corners[0]]])
            #plane_holonomies: gives array of each matrix corresponding to the holonomy around a plaquette
            #in list corresponding to posiiton of corner node in self.real_nodearray
            plane_holonomies = Blist[:, np.newaxis, np.newaxis] *  ((first_link_matricies @ second_link_matricies @ third_link_matricies @ fourth_link_matricies))
            node_action_contribs += np.sum(np.trace(plane_holonomies, axis1 = 1, axis2 = 2))
        action = 2 * len(self.planeslist) * self.num_nodes - node_action_contribs

        #calculating fictuous momentum contribution
        momentum_contrib = np.trace(np.sum(momentum_array@momentum_array, axis = 0))

        #returning Hamiltonian value
        hamiltonian = momentum_contrib + action
        print("elapsed:", time.time()-starttime)
        return np.real(hamiltonian)



    def accept_config(self, new_configuration, initial_configuration):
        starttime = time.time()
        new_configuration = self.ghost_fill_configuration(new_configuration)
        Hinitial = self.hamiltonian(initial_configuration)
        Hnew = self.hamiltonian(new_configuration)
        difference = Hnew - Hinitial
        transition_prob = np.minimum(1, np.exp(-difference))
        randomvar = random.uniform(0,1)
        print("Hnew: ", Hnew, " Hinitial: ",Hinitial)
        if randomvar < transition_prob:
            new_matrix_dict = dict(zip(self.link_dict.keys(), new_configuration[0].values()))
            for node in self.real_nodearray:
                for direction in range(self.dimensions):
                    self.link_dict[node][direction].set_matrix(new_matrix_dict[node][direction])
            print("overall elapsed:", time.time() - starttime)
            return True
        else:
            print("overall elapsed:", time.time() - starttime)
            return False


    def chain(self, number_iterations):
        momentum =self.random_momentum()
        for i in range(number_iterations):
            print(i)
            old_config = [self.get_link_matrix_dict(), momentum]
            candidate = self.generate_candidate_configuration(old_config, 5)
            if self.accept_config(old_config, candidate):
                momentum = candidate[1]
            else:
                continue
        return





