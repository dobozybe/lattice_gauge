import numpy as np
from node_links import *
import random
import matplotlib.pyplot as plt
import time
import scipy.stats


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
    def __init__(self, shape):
        # dimensions should be an nd numpy array with each entry corresponding to the size of the array.
        # t should be the first dimension
        starttime = time.time()
        self.shape = shape
        self.twists = {}
        self.paddedshape = []
        self.planeslist = []
        self.indexdict ={}
        self.num_nodes = 1 #not including ghosts
        for i in shape:
            self.num_nodes *= i


        for i in range(len(shape)): #make array bigger along each axis by one node. Outer padded layer is "ghost links"
            self.paddedshape.append(shape[i]+1)

        self.twistGenFunc = get_identity
        self.dimensions = len(shape)
        self.nodelist = []
        tuplelist = []
        self.make_tuples(np.zeros(len(shape)), self.paddedshape, tuplelist)


        for coordinates in tuplelist: #initialize nodes and add them to nodelist
            self.nodelist.append(Node(coordinates, is_ghost_node= [self.is_boundary(coordinates, i) for i in range(self.dimensions)]))#loops will ignore ghost nodes. If it's on the boundary it is a ghost node
            self.indexdict[tuple(coordinates)] = self.get_node_index(coordinates)
        for i in range(self.dimensions): #initialize planeslist
            for j in range(i):
                self.planeslist.append([j,i])

        self.make_links()
        self.make_boundary_links()
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
        return self.nodelist[self.indexdict[tuple(coordinates)]]

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
        for node in self.nodelist:
            for i in range(len(self.shape)):
                these_coords = node.coordinates
                max_node = self.shape[i]
                if these_coords[i] < max_node:
                    #if we're not in the padded part of the array (not a ghost node) link to the next node
                    new_coords = np.concatenate(
                        (these_coords[:i], [(these_coords[i] + 1)], these_coords[i + 1:]))
                    newlink=Link(node, self[new_coords], i)
                    newlink.set_matrix(randomSU2())
                else:
                    pass
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
                        identified_node_coordinates[i] = 0
                        identified_link = self[identified_node_coordinates].get_link(j, 0)
                        identified_link_matrix = identified_link.get_matrix()
                        new_link.set_matrix( #todo: implement twists. Right now this is periodic BCs.
                            np.diag([1,1]) @ identified_link_matrix
                        )
                        new_link.set_parent(identified_link)

    def get_plaquette_corners(self, node, plane): #with corner as bottom left point in plaquette
        cornercoords = node.coordinates
        first_corner = cornercoords
        second_corner = self[cornercoords].get_next_node(plane[0], 0).coordinates
        third_corner = self[second_corner].get_next_node(plane[1], 0).coordinates
        fourth_corner = self[third_corner].get_next_node(plane[0], 1).coordinates
        return [first_corner, second_corner, third_corner, fourth_corner]

    def get_plaquette_links(self, node, plane): #ordered in correct orientation
        corners = self.get_plaquette_corners(node, plane)
        first_corner = corners[0]
        second_corner = corners[1]
        third_corner = corners[2]
        fourth_corner = corners[3]
        linklist = [
            self[first_corner].get_link(plane[0], 0),
            self[second_corner].get_link(plane[1], 0),
            self[third_corner].get_link(plane[0], 1),
            self[fourth_corner].get_link(plane[1], 1)
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
        start_time = time.time()
        cornernode = self[cornercoords]
        product = np.diag([1,1])
        for matrix in self.get_plaquette_matricies(cornernode,plane):
            product = product @ matrix
        return product


    def get_action(self):
        starttime = time.time()
        action_sum = 0
        loop_count = 0
        num_plaquettes = len(self.planeslist) * self.num_nodes
        for node in self.nodelist:
            if (not (True in node.ghost_node)):
                for plane in self.planeslist:
                    loop_count +=1
                    start_time = time.time()
                    action_sum+= np.trace(self.get_plaquette_holonomy(node.coordinates, plane)) #2.2 of FDW paper
            else:
                pass
        action = 2 * num_plaquettes - action_sum
        if np.imag(action) > 0.0001:
            print("Warning: Action is Complex")
            return action
        #print("action found in ", time.time()-starttime)
        return np.real(action)
        """
        if np.imag(action) == 0:
            return np.real(action)
        else:
            return "Error: Action has imaginary part"
        """

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

                Vdagmatrix_node = (self.translate(node, link_direction, 1).get_link(orthogonal_direction, 0).get_matrix()
                                @ self.translate(node, orthogonal_direction, 1).get_link(link_direction, 0).get_matrix().conj().T
                                @ node.get_link(orthogonal_direction, 0).get_matrix().conj().T) #formula worked out in notebook (twice. This is verified).
                Vdagmatrix_complement = (self.twistGenFunc(self.translate(node, link_direction, 1), orthogonal_direction).conj().T
                                      @self.translate(complement_node, link_direction, 1).get_link(orthogonal_direction, 0).get_matrix().conj().T
                                      @ complement_node.get_link(link_direction,0).get_matrix().conj().T
                                      @complement_node.get_link(orthogonal_direction, 0).get_matrix()
                                      @self.twistGenFunc(node, orthogonal_direction)
                                      )
                Vdagsum += Vdagmatrix_node
                Vdagsum += Vdagmatrix_complement
        Vdagsum[0][1] = -np.conj(Vdagsum[1][0]) #correcting entries of Vdagsum so that it doesn't drift from a real number times an SU2 matrix
        Vdagsum[1][1] = np.conj(Vdagsum[0][0])
        if np.linalg.det(Vdagsum) == 0:
            print("Division by zero encountered")
            print(Vdagsum)
            return
        original_contrib = -np.trace(link.get_matrix() @ Vdagsum) #trace (U times vdag) is the original contribution to the action
        rsquared = np.real(np.linalg.det(Vdagsum))
        Wdag = Vdagsum/(np.sqrt(rsquared)) #since Wdag = Vsum/r, r = sqrt(det(Vsum)) (since det(Wdag = 1)). Then we can get Wdag from Vsum.
        new_contrib = - 2 * np.sqrt(rsquared) #should be equal to -2r
        #print("calculated:", new_contrib, " should be:", -2*np.sqrt(rsquared))
        if not inSU2(Wdag):
            print("Warning! No longer in SU(2)!")
            return 1
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
        print("Sweeping")
        for direction in range(len(self.shape)):
            for node in self.nodelist:
                if not (True in node.ghost_node):
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

    def action_min_sweep(self, nsweeps): #1 sweep down to 7 seconds
        actionlist = np.zeros(nsweeps + 1)
        start_action = self.get_action()
        actionlist[0] += start_action
        print("Starting Action:", actionlist[0])
        for i in range(nsweeps):
            new_action = self.lattice_sweep(actionlist[i])
            actionlist[i+1] += new_action
            print("Calculated action after sweep:", actionlist[i+1])
        plt.plot(actionlist, marker = "o", linestyle = "")
        plt.show()
        return actionlist




