import numpy as np
from node_links import *
import random
import matplotlib.pyplot as plt
import time
import pickle
from sympy import LeviCivita
import multiprocessing.dummy as mp
from functools import partial
import itertools




g = 1


hamiltonian_list = []
action_list = []
momentum_list = []
momentum_change_list=[]
action_change_list=[]



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


def get_identity():
    return np.diag(np.full(2, 1))


lie_gens = np.stack([ #i sigma_n where sigma_i are the paulis.
    np.array([[0, 1], [-1, 0]]),
    np.array([[0, 1j], [1j, 0]]),
    np.array([[1j, 0], [0, -1j]])
])


def su2_exp(matrixarray):
    array_shape = np.shape(matrixarray)
    matrixarray = matrixarray.reshape(-1,2,2)
    gen0 = np.broadcast_to(lie_gens[0].conj().T, matrixarray.shape)
    gen1 = np.broadcast_to(lie_gens[1].conj().T, matrixarray.shape)
    gen2 = np.broadcast_to(lie_gens[2].conj().T, matrixarray.shape)
    param1 = np.trace(matrixarray @ gen0, axis1=1, axis2=2)/2
    param2 = np.trace(matrixarray @ gen1, axis1=1, axis2=2)/2
    param3 = np.trace(matrixarray @ gen2, axis1=1, axis2=2)/2

    paramarray = np.stack([param1, param2, param3]).T
    normparams = np.sqrt(param1 ** 2 + param2 ** 2 + param3 ** 2)
    if np.max(np.imag(normparams))*(g**2) > 0.01:
        print("error! Normparams in su2_exp is imaginary!", np.max(np.imag(normparams)))
        return None
    normparams = np.float64(normparams)
    modded_normparams = np.mod(normparams, 2 * np.pi)
    normparams_safe = np.where(normparams == 0, 1e-12, normparams)
    normed_paramarray = paramarray/normparams_safe[:, np.newaxis]
    exp = np.eye(2)[np.newaxis, :, :] * np.cos(modded_normparams)[:, np.newaxis, np.newaxis]+np.einsum(
        "ij, jkl->ikl", normed_paramarray, lie_gens
    ) * np.sin(modded_normparams)[:, np.newaxis, np.newaxis]
    exp = exp.reshape(array_shape)
    return exp


def random_su2_matrix():
    # Random real numbers for the basis coefficients
    a1, a2, a3 = np.random.randn(3)

    # su(2) basis using Pauli matrices (times i)
    sigma1 = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma3 = np.array([[1, 0], [0, -1]], dtype=complex)

    # Linear combination: X = i(a1 * σ1 + a2 * σ2 + a3 * σ3)
    su2_matrix = 1j * (a1 * sigma1 + a2 * sigma2 + a3 * sigma3)
    return su2_matrix


x = np.array([random_su2_matrix()])




def randomSU2():
    alpha = random.random() * 2 * np.pi
    beta = random.random() * 2 * np.pi
    gamma = random.random() * 2 * np.pi
    a = np.exp(alpha * 1j) * np.sin(beta)
    b = np.exp(gamma * 1j) * np.cos(beta)
    matrix = np.array([[a, -np.conj(b)], [b, np.conj(a)]])
    return matrix


def inSU2(matrix, tol=1e-10):
    det_close = np.allclose(np.linalg.det(matrix), 1.0, atol=tol)
    unitary = np.allclose(matrix @ matrix.conj().T, np.eye(2), atol=tol)
    return det_close and unitary


def project(matrixarray):
    antisym = (matrixarray - np.transpose(matrixarray.conj(), axes=(*range(matrixarray.ndim - 2), -1, -2)))
    return (1 / 2) * antisym - (1 / 4) * np.trace(antisym, axis1 = -2, axis2 = -1)[...,np.newaxis, np.newaxis] * np.array([[1, 0], [0, 1]])

def _momentum_update_helper(config, dt, coordinates, dimensions, Bdict, V2Bdict, staple_dict):
    new_matrix_list = list(range(dimensions)) #creates a list with the right number of indicies
    link_dict = config[0]
    momentum_dict = config[1]
    node_momenta = momentum_dict[coordinates]
    for mu in range(dimensions):
        Vmu =0
        for nu in range(dimensions):
            if mu != nu:
                staple_links = staple_dict[(coordinates, mu,nu)]
                Bval = Bdict[coordinates, (mu, nu)]
                V2Bval = V2Bdict[coordinates, (mu, nu)]
                first_staple = Bval * staple_links[0][0] @ staple_links[0][1].conj().T @ staple_links[0][2].conj().T
                second_staple = V2Bval * staple_links[1][0].conj().T @ staple_links[1][1].conj().T @  staple_links[1][2]
                Vmu +=first_staple + second_staple
        stapleterm = link_dict[coordinates][mu] @ Vmu
        momentum_change = (1/g**2) * (stapleterm.conj().T - stapleterm) * dt
        new_matrix_list[mu] = node_momenta[mu] + momentum_change
    return {coordinates:tuple(new_matrix_list)}

def carg_momentum_update_helper(config, dt, coordinates, dimensions, Bdict, V2Bdict,
                        staple_dict):  # sends momentum at n*dt - dt/2 to n * dt + dt/2
    starttime = time.time()
    momentum_dict = config[1].copy()
    link_dict = config[0]
    link_array = np.transpose(np.array(list(config[0].values())), axes=[1, 0, 3, 2])

    # link matricies dicts have indicies [direction, plane, node, matrix]

    # do it slowly to make sure you're implementing EOM correctly. This conserves energy it's just slow as shit
    old_momentum_list = momentum_dict[coordinates].copy()
    for direction in range(dimensions):
        Vfirst = 0
        Vsecond = 0
        for sum_direction in range(dimensions):
            if sum_direction != direction:
                staple_links = staple_dict[(coordinates, direction,sum_direction)]
                Vfirst += Bdict[coordinates, (direction, sum_direction)] * staple_links[0][0] @ staple_links[0][
                    1].conj().T @ staple_links[0][2].conj().T

                Vsecond += V2Bdict[coordinates, (direction, sum_direction)] * staple_links[1][0].conj().T @ \
                           staple_links[1][1].conj().T @ staple_links[1][2]

        Vdirection = Vfirst + Vsecond
        momentum_change = (1 / g ** 2) * (
                    Vdirection.conj().T @ link_dict[coordinates][direction].conj().T - link_dict[coordinates][
                direction] @ Vdirection)

        old_momentum_list[direction] += momentum_change * dt

    return {coordinates: old_momentum_list}


















class Lattice:  # ND torus lattice


    """INITIALIZATION METHODS"""


    def __init__(self, shape, filename=None, twistmatrix=None):
        # dimensions should be an nd numpy array with each entry corresponding to the size of the array.
        # t should be the first dimension
        starttime = time.time()
        self.SUN_dimension = 2
        self.shape = shape
        self.filename = filename
        self.processes = 8
        self.plaquette_corner_dict = {}
        self.v_second_plaquette_corner_dict = {}
        self.twistmatrix = np.zeros((len(shape), len(shape)))
        if twistmatrix is not None:
            self.twistmatrix = twistmatrix
        self.twists = []
        for i in range(np.shape(self.twistmatrix)[0]):
            for j in range(np.shape(self.twistmatrix)[1]):
                if j>i and self.twistmatrix[i][j] ==1:
                    self.twists.append([i,j])

        self.paddedshape = []
        self.planeslist = []

        self.indexdict = {}
        self.index_lookup_dict = {}
        self.Bdict = {}
        self.V2_Bdict = {}
        self.link_dict = {}
        self.chunksize = 1


        self.num_nodes = 1  # not including ghosts
        for i in shape:
            self.num_nodes *= i

        for i in range(
                len(shape)):  # make array bigger along each axis by one node. Outer padded layer is "ghost links"
            self.paddedshape.append(shape[i] + 1)

        self.twistGenFunc = get_identity
        self.dimensions = len(shape)
        self.nodelist = []
        self.real_nodearray = []  # np array of nodes that aren't ghost nodes
        self.ghost_nodearray = []
        tuplelist = []
        self.make_tuples(np.zeros(len(shape), dtype=int), self.paddedshape, tuplelist)

        for coordinates in tuplelist:  # initialize nodes and add them to nodelist
            self.nodelist.append(Node(coordinates, is_ghost_node=[self.is_boundary(coordinates, i) for i in range(
                self.dimensions)]))  # loops will ignore ghost nodes. If it's on the boundary it is a ghost node
            self.indexdict[tuple(coordinates)] = self.nodelist[self.get_node_index(coordinates)]
            self.index_lookup_dict[tuple(coordinates)]=tuple(coordinates)

        for node in self.nodelist:  # initialize real node array
            if not True in node.ghost_node:
                self.real_nodearray.append(node)
                node.index_in_real_nodearray = self.real_nodearray.index(node)
            else:
                self.ghost_nodearray.append(node)
        self.real_nodearray = np.array(self.real_nodearray)
        self.ghost_nodearray = np.array(self.ghost_nodearray)

        for i in range(self.dimensions):  # initialize planeslist
            for j in range(i):
                self.planeslist.append(tuple([j, i]))
        self.make_links()
        self.make_boundary_links()

        for node in self.nodelist:
            self.link_dict[node] = np.array(node.links).T[0]

        """for node in self.real_nodearray:
            for plane in self.planeslist:
                self.plaquette_corner_dict[(node, plane)] = self.get_plaquette_corners(node, plane)
                self.Bdict[(node, plane)] = self.B(plane[0], plane[1], node)"""

        self.staple_array = np.empty((len(self.real_nodearray), self.dimensions, self.dimensions, 2, 3), dtype = object)
        self.Barray = np.empty((len(self.real_nodearray), self.dimensions, self.dimensions, 1, 1), dtype = np.complex64)
        self.V2_Barray =np.empty((len(self.real_nodearray), self.dimensions, self.dimensions, 1, 1), dtype = np.complex64)
        for nodeindex,node in enumerate(self.real_nodearray):
            for mu in range(self.dimensions):
                for nu in range(self.dimensions):
                    if nu != mu:
                        plane = (mu,nu)
                        self.plaquette_corner_dict[(node, plane)] = self.get_plaquette_corners(node, plane)
                        self.v_second_plaquette_corner_dict[(node,plane)] = self.v2_second_plaquette_corners(node, plane)
                        self.Bdict[(node.tuplecoords, plane)] = self.B(plane[0], plane[1], node)
                        self.V2_Bdict[(node.tuplecoords, plane)] = self.B(plane[0], plane[1], self.translate(node, plane[1], -1))

                        self.Barray[nodeindex, mu, nu, 0,0] = self.B(mu, nu, node)
                        self.V2_Barray[nodeindex, mu, nu, 0, 0] = self.B(mu, nu, self.translate(node, nu, -1))

                        #setting staple matricies
                        self.staple_array[nodeindex, mu, nu, 0,0] = self.translate(node, mu, 1).get_link(nu, 0)
                        self.staple_array[nodeindex, mu, nu, 0, 1] = self.translate(node, nu, 1).get_link(mu, 0)
                        self.staple_array[nodeindex, mu, nu, 0, 2] = node.get_link(nu, 0)
                        self.staple_array[nodeindex, mu, nu, 1, 0] = self.translate(self.translate(node, mu, 1), nu, -1).get_link(nu, 0)
                        self.staple_array[nodeindex, mu, nu, 1, 1] = self.translate(node, nu, -1).get_link(mu, 0)
                        self.staple_array[nodeindex, mu, nu, 1, 2] = self.translate(node, nu, -1).get_link(nu, 0)
                    elif nu == mu:
                        plane = (mu,nu)
                        self.Barray[nodeindex, mu, nu, 0, 0] = self.B(mu, nu, node)
                        self.V2_Barray[nodeindex, mu, nu, 0, 0] = self.B(mu, nu, self.translate(node, nu, -1))
                        self.Bdict[(node.tuplecoords, plane)] = self.B(plane[0], plane[1], node)
                        self.V2_Bdict[(node.tuplecoords, plane)] = self.B(plane[0], plane[1], self.translate(node, plane[1], -1))
                        self.staple_array[nodeindex, mu, nu, 0, 0] = None
                        self.staple_array[nodeindex, mu, nu, 0, 1] = None
                        self.staple_array[nodeindex, mu, nu, 0, 2] = None
                        self.staple_array[nodeindex, mu, nu, 1, 0] = None
                        self.staple_array[nodeindex, mu, nu, 1, 1] = None
                        self.staple_array[nodeindex, mu, nu, 1, 2] = None

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

    def translate(self, node, direction,
                  distance):  # given a node, return the node distance units later in a given direction
        position = node.coordinates.copy()
        position[direction] = (position[direction] + distance) % self.shape[direction]
        return self[position]

    def make_tuples(self, tuple, depths, outputlist):
        # generates a list of tuples in the nd array. Equivalent of "range" essentially. Only used to populate a dictionary during init
        if depths == []:
            outputlist.append(tuple.copy())
            return
        for j in range(depths[0]):
            tuple[-len(depths)] = j
            self.make_tuples(tuple, depths[1:], outputlist)

    def is_boundary(self, coordinates, direction):  # todo: this feels inelegant
        if coordinates[direction] == self.shape[direction]:
            return True
        else:
            return False

    def make_links(self):
        if self.filename != None:
            datafile = open(self.filename + ".pickle", "rb")
            matrixdict = pickle.load(datafile)
        for node in self.real_nodearray:
            # if we're not in the padded part of the array (not a ghost node) link to the next node in each direction
            for i in range(len(self.shape)):
                these_coords = node.coordinates
                new_coords = np.concatenate(
                    (these_coords[:i], [(these_coords[i] + 1)], these_coords[i + 1:]))
                newlink = Link(node, self[new_coords], i)
                if self.filename == None:
                    newlink.set_matrix(randomSU2())
                else:
                    this_matrix = matrixdict[str(these_coords) + ":" + str(i)]
                    newlink.set_matrix(this_matrix)
        return

    def make_student_link_adjacent(self, node, direction):
        next_coords = np.concat(
            (node.coordinates[:direction], [node.coordinates[direction] + 1], node.coordinates[direction + 1:]))
        new_link = StudentLink(node, self[next_coords], direction)
        return new_link

    def make_boundary_links(self):  # todo: clean this up
        for i in range(self.dimensions):
            for node in self.nodelist:
                for j in range(self.dimensions):
                    if (node.ghost_node[i] and (not node.ghost_node[
                        j])):  # don't add links in the i direction (we're at the i boundary of the padded lattice)
                        new_link = self.make_student_link_adjacent(node, j)
                        identified_node_coordinates = node.coordinates.copy()
                        for k in range(self.dimensions):
                            identified_node_coordinates[k] = identified_node_coordinates[k] % self.shape[k]
                        identified_link = self[identified_node_coordinates].get_link(j, 0)
                        identified_link_matrix = identified_link.get_matrix()
                        # print(identified_link)
                        new_link.set_matrix(identified_link_matrix)
                        new_link.set_parent(identified_link)

    def B(self, mu, nu, node):
        position = node.coordinates
        if (position[mu] == self.shape[mu] - 1) and (position[nu] == self.shape[nu] - 1):
            return np.exp(-2 * np.pi * 1j * float(self.twistmatrix[mu][nu]) / self.SUN_dimension)
        else:
            return float(1)

    def save_links(self, filename):
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


    """UTILITY METHODS"""

    def get_link_matrix_dict(self):  # returns only non-ghost link matricies
        matrix_dict = {}
        for node in self.real_nodearray:
            matrixholder = []
            for i in range(self.dimensions):
                thislink = node.get_link(i, 0)
                if thislink != None:
                    matrixholder.append(thislink.get_matrix())
                else:
                    matrixholder.append(np.array([[None, None], [None, None]]))
            matrix_dict[node.tuplecoords] = matrixholder
        return matrix_dict

    def get_plaquette_corners(self, node, plane):  # with corner as bottom left point in plaquette
        cornercoords = node.coordinates
        first_corner = node
        second_corner = first_corner.get_next_node(plane[0], 0)
        third_corner = second_corner.get_next_node(plane[1], 0)
        fourth_corner = third_corner.get_next_node(plane[0], 1)
        return [first_corner, second_corner, third_corner, fourth_corner]

    def v2_second_plaquette_corners(self, node, plane):  # with corner as bottom left point in plaquette
        first_corner = node
        second_corner = self.translate(first_corner, plane[0], 1)
        third_corner = self.translate(second_corner, plane[1], -1)
        fourth_corner = self.translate(third_corner, plane[0], -1)
        return [first_corner, second_corner, third_corner, fourth_corner]

    def get_plaquette_links(self, node, plane):  # ordered in correct orientation
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

    def get_plaquette_matricies(self, node, plane):
        links = self.get_plaquette_links(node, plane)
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
        product = np.diag([1, 1])
        for matrix in self.get_plaquette_matricies(cornernode, plane):
            product = product @ matrix
        return product

    def get_action(self):
        action_sum = 0
        loop_count = 0
        num_plaquettes = len(self.planeslist) * self.num_nodes
        for node in self.real_nodearray:
            # print(node.coordinates)
            for plane in self.planeslist:
                loop_count += 1
                action_sum += np.trace(self.B(plane[0], plane[1], node) * self.get_plaquette_holonomy(node.coordinates,
                                                                                                      plane))  # 2.2 of FDW paper
        action = 2 * num_plaquettes - action_sum
        action = (2/g**2) * action
        if np.imag(action) > 0.0001:
            print("Warning: Action is Complex")
            return action
        return np.real(action)

    def ghost_fill_configuration(self, configuration):
        config_link_dict = configuration[0]
        momentum_dict = configuration[1]
        filled_link_dict = config_link_dict.copy()
        filled_momentum_dict = momentum_dict.copy()
        for node in self.ghost_nodearray:
            linkholder = []
            momentumholder = []
            for i in range(self.dimensions):
                if not node.ghost_node[i]:
                    linkholder.append(config_link_dict[node.get_link(i, 0).parent_link.node1.tuplecoords][
                                          i])  # I've already done the math about which links should be identified
                    momentumholder.append(momentum_dict[node.get_link(i, 0).parent_link.node1.tuplecoords][
                                              i])
                else:
                    linkholder.append(np.array([[None, None], [None,
                                                               None]]))  # none array should never be involved in any calculation but should crash if it somehow is.
                    momentumholder.append(np.array([[None, None], [None,
                                                                   None]]))
            filled_link_dict[node.tuplecoords] = linkholder
            filled_momentum_dict[node.tuplecoords] = momentumholder
        new_config = [filled_link_dict, filled_momentum_dict]
        return new_config

    def measure_observables(self,observables):
        observable_dict = {}
        for observable in observables:
            observable_dict[observable.identifier] = tuple(observable.evaluate(self))
        return observable_dict

    """UTILITY FOR PICKLING"""

    """def __getstate__(self):
        return {"shape":self.shape,
                "twists":self.twistmatrix,
                "index_lookup_dict":self.index_lookup_dict,
                "processes":self.processes
                }

    def __setstate__(self, state):
        """


    """COOLING METHODS"""

    def minimize_link_action(self, link): #doesn't work for theoretical reasons for a 1x1x1x1 lattice!
        """Find all plaquettes the link contributes to"""
        Vdagsum = np.full((2, 2), 0 + 0j)  # for SU(2)
        # print("Start action:", start_action)
        for plane in self.planeslist:
            if link.direction in plane:
                #this_plane = plane.copy()
                this_plane = [plane[0],plane[1]]
                link_direction = link.direction
                this_plane.remove(link_direction)
                orthogonal_direction = this_plane[0]  # gives direction in plane orthogonal to link
                node = link.node1
                complement_node = self.translate(node, orthogonal_direction,
                                                 -1)  # other node whose plaquette in that plane will contain that link
                # This logic is good (checked thrice)
                Vdagmatrix_node = (self.B(plane[0], plane[1], node) * self.translate(node, link_direction, 1).get_link(
                    orthogonal_direction, 0).get_matrix()\
                                   @ self.translate(node, orthogonal_direction, 1).get_link(link_direction,
                                                                                            0).get_matrix().conj().T\
                                   @ node.get_link(orthogonal_direction,
                                                   0).get_matrix().conj().T)  # formula worked out in notebook (twice. This is verified).
                """Vdagmatrix_complement = (
                            np.conjugate(self.B(plane[0], plane[1], complement_node)) * self.translate(complement_node,
                                                                                                       link_direction,
                                                                                                       1).get_link(
                        orthogonal_direction, 0).get_matrix().conj().T
                            @ complement_node.get_link(link_direction, 0).get_matrix().conj().T
                            @ complement_node.get_link(orthogonal_direction, 0).get_matrix()
                            )"""
                Vdagmatrix_complement = self.B(plane[0],plane[1], complement_node) *self.translate(self.translate(node, link_direction, 1), orthogonal_direction, -1).get_link(orthogonal_direction, 0).get_matrix().conj().T\
                                        @ complement_node.get_link(link_direction, 0).get_matrix().conj().T\
                                        @ complement_node.get_link(orthogonal_direction, 0).get_matrix()

                Vdagsum += Vdagmatrix_node
                Vdagsum += Vdagmatrix_complement
        Vdagsum[0][1] = -np.conj(Vdagsum[1][
                                     0])  # correcting entries of Vdagsum so that it doesn't drift from a real number times an SU2 matrix
        Vdagsum[1][1] = np.conj(Vdagsum[0][0])
        rsquared = np.real(np.linalg.det(Vdagsum))
        original_contrib = -(2/g**2) * np.trace(
            link.get_matrix() @ Vdagsum)  # trace (U times vdag) is the original contribution to the action

        Wdag = Vdagsum / (np.sqrt(
            rsquared))  # since Wdag = Vsum/r, r = sqrt(det(Vsum)) (since det(Wdag = 1)). Then we can get Wdag from Vsum.
        new_contrib = - (4/g**2) * np.sqrt(rsquared)  # should be equal to -2r
        # print("calculated:", new_contrib, " should be:", -2*np.sqrt(rsquared))
        """if not inSU2(Wdag):
            print("Warning! No longer in SU(2)!")
            return 1"""
        link.set_matrix(
            Wdag.conj().T)  # set the matrix of our link to the dagger of Wdag. This should minimize this link's contribution to the action per FDW paper.
        # It should also deal with the reverse link matrix (since the link *is* the reverse link)
        """if link.student_links != []:  # update the student if there is one
            for student_link in link.student_links:
                student_link.update()"""
        action_delta = new_contrib - original_contrib
        return np.real(action_delta)

    def lattice_sweep(self, initial_action):
        starttime = time.time()
        actionchanges = []
        action = initial_action
        original_action_changes = []
        contribution_counter = 0
        # print("Sweeping")
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
        print("Sweep Completed in", time.time() - starttime)
        return action

    def action_min_sweep(self, nsweeps):  # 1 sweep down to ~6 seconds. 1000 sweeps about 1.5 hours.
        start_time = time.time()
        actionlist = np.zeros(nsweeps + 1)
        start_action = self.get_action()
        actionlist[0] += start_action
        print("Starting Action:", actionlist[0])
        for i in range(nsweeps):
            print("Sweep ", i)
            new_action = self.lattice_sweep(actionlist[i])
            actionlist[i + 1] += new_action
            print("Calculated action after sweep:", actionlist[i + 1])
        plt.plot(actionlist, marker="o", linestyle="")
        print("total time:", time.time() - start_time)
        #plt.show()
        return actionlist

    def action_density_plot(self, plane,
                            planecoords):  # plane coords should be other coordinates in order with the plane coordinates removed
        xlist = range(self.shape[plane[0]])
        ylist = range(self.shape[plane[1]])
        actionlist = np.zeros((self.shape[plane[1]], self.shape[plane[0]]))
        for node in self.real_nodearray:
            if (np.delete(np.delete(node.coordinates, plane[1]), plane[0]) == planecoords).all():
                this_location = [node.coordinates[plane[0]], node.coordinates[plane[1]]]
                this_action = 2 - np.trace(
                    self.B(plane[0], plane[1], node) * self.get_plaquette_holonomy(node.coordinates, plane))
                actionlist[int(this_location[1])][int(this_location[0])] = np.real(
                    this_action)  # first index selects row and second selects column
        plt.pcolormesh(xlist, ylist, actionlist)
        titlestring = "with x"
        skipped = 0
        for i in range(self.dimensions):
            if i in plane:
                skipped += 1
                pass
            else:
                if i != self.dimensions - 1:
                    titlestring += str(i) + " = " + str(planecoords[i - skipped]) + ", x"
                else:
                    titlestring += str(i) + " = " + str(planecoords[i - skipped])
        plt.legend()
        plt.title("Action Density in the " + str(plane) + " plane, with " + titlestring)
        plt.show()

    def cooling_measure(self, iterations, observables, coolingsteps = 1000, log=True, filename=None):
        observable_list = []

        for i in range(iterations):
            self.action_min_sweep(coolingsteps)
            observable_list.append(self.measure_observables(observables))
            self.__init__(self.shape, twistmatrix=self.twistmatrix)

        if (log == True or filename != None):
            if filename != None:
                with open("Cooling " + filename + str([obs.identifier for obs in observables]) + str(
                        self.shape) + ".txt", "w") as f:
                    for item in observable_list:
                        f.write(f"{item}\n")
            else:
                with open("Cooling " + str(time.strftime("%d-%m-%Y_%H:%M:%S")) + str(
                        [obs.identifier for obs in observables]) + str(
                        self.shape) + ".txt", "w") as f:
                    for item in observable_list:
                        f.write(f"{item}\n")
    def get_topological_charge(self):
        runningsum = 0
        indicies = [[i, j, k, l] for i in range(3) for j in range(3) for k in range(3) for l in range(3)]
        for node in self.real_nodearray:
            for index in indicies:
                civita = LeviCivita(index[0], index[1], index[2], index[3])
                if civita == 0:
                    continue
                contribution = civita * np.trace(
                    self.get_plaquette_holonomy(node.coordinates, [index[0], index[1]])
                    @ self.get_plaquette_holonomy(node.coordinates, [index[2], index[3]])
                )
                runningsum += contribution
        return -1 / (32 * np.pi ** 2) * runningsum

    """HYBRID MONTE CARLO FUNCTIONS. 
    Will repeatedly use variable name "configuration" to represent a pair
     [link variable matrix dict, computer momentum dict]. It's important that the first entry
     is link variable matricies, and not the link variable objects themselves so we don't reset them
     during time evolution. Dicts are formatted as dict[node's coords] = [list of matricies indexed by direction] Dictionaries only contain real nodes, there's a method to ghost fill dictionaries when
     necessary."""



    def random_momentum(self):
        new_momentum_dict = {}
        imaglist = []
        for node in self.real_nodearray:
            momenta_list = []
            for direction in range(self.dimensions):
                momenta_vals = np.array([np.random.normal(scale = 1/np.sqrt(2)), np.random.normal(scale = 1/np.sqrt(2)), np.random.normal(scale = 1/np.sqrt(2))])
                matrix = momenta_vals[0] * lie_gens[0] + momenta_vals[1] * lie_gens[1] + momenta_vals[2] * lie_gens[2]
                momenta_list.append(
                    momenta_vals[0] * lie_gens[0] + momenta_vals[1] * lie_gens[1] + momenta_vals[2] * lie_gens[2])
                imaglist.append(np.array([np.imag(np.trace(matrix@lie_gens[0])),np.imag(np.trace(matrix@lie_gens[1])),np.imag(np.trace(matrix@lie_gens[2]))]))
            new_momentum_dict[node.tuplecoords] = momenta_list
        imaglist = np.sum(np.abs(np.array(imaglist)), axis = 1)
        return new_momentum_dict



    def link_update(self, initial_config, dt):  # sends link at n * dt to (n+1)*dt
        link_array = np.transpose(np.array(list(initial_config[0].values())), axes=[1, 0, 2, 3])
        newlinks = []
        momentum_array = np.transpose(np.array(list(initial_config[1].values())), axes=[1, 0, 2, 3])
        for direction in range(self.dimensions):
            new_link_array = su2_exp(dt * momentum_array[direction]) @ link_array[direction]
            newlinks.append(new_link_array)
        newlinks = np.transpose(np.array(newlinks), axes=[1, 0, 2, 3])

        new_link_dict = dict(zip(initial_config[0].keys(), list(newlinks)))
        new_config = [new_link_dict, initial_config[1]]
        return new_config

    def single_node_link_update(self, initial_config_arrays, dt, node_index):
        link_array = initial_config_arrays[0]
        momentum_array = initial_config_arrays[1]
        new_link_array = su2_exp(dt * momentum_array[node_index]) @ link_array[node_index] #should give a list of matricies indexed by direction
        return new_link_array

    def batched_single_node_link_update(self, initial_config_arrays, dt, start, end):
        link_array = initial_config_arrays[0]
        momentum_array = initial_config_arrays[1]

        batch_link = link_array[start:end+1]
        batch_momentum = momentum_array[start:end+1]

        batch_out = su2_exp(dt * batch_momentum) @ batch_link
        return batch_out


    def parallel_link_update(self, initial_config_arrays, dt):
        print(np.shape(initial_config_arrays))
        inputs = [[initial_config_arrays, dt, nodeindex] for nodeindex in range(len(self.real_nodearray))]
        pool = self._pool
        new_links_array = np.stack(pool.starmap(self.single_node_link_update, inputs, chunksize = self.chunksize))
        return np.stack([new_links_array, initial_config_arrays[1]])




    def single_node_momentum_update(self,initial_config_array,dt,node_index,staple_matrix_array):

        link_array = initial_config_array[0]
        momentum_array = initial_config_array[1]

        staple_matricies = staple_matrix_array[node_index]


        #indicies are node, direction mu, direction nu, first/second term, list of staple matricies


        #extract staple matricies
        firststaplematricies, secondstaplematricies = np.split(staple_matricies, 2, axis = 2)


        staple11, staple12, staple13 = np.split(firststaplematricies, 3, axis = 3)
        staple21, staple22, staple23 = np.split(secondstaplematricies, 3, axis=3)


        # calculate the twisted staple for each nu (index 2)
        firststaple = self.Barray[...,None,None][node_index] * staple11 @ staple12.conj().swapaxes(-1,-2) @ staple13.conj().swapaxes(-1,-2)
        secondstaple = self.V2_Barray[...,None, None][node_index] * staple21.conj().swapaxes(-1,-2) @ staple22.conj().swapaxes(-1,-2) @ staple23


        #adding two terms together to get full staple then summing to get Vmu array. Squeeze to get rid of vestigal indicies
        staplesum = firststaple + secondstaple

        Varray = np.sum(staplesum, axis = 1)
        Varray = np.squeeze(Varray)

        #calculating momentum update
        stapleterm = link_array[node_index] @ Varray
        momentum_change = (1/g**2) * (stapleterm.conj().swapaxes(-1,-2) - stapleterm) * dt

        new_momentum_array = momentum_array[node_index] + momentum_change


        new_momentum_array = np.squeeze(new_momentum_array)
        #print(np.shape(new_momentum_array))

        #new_momentum={coordinates:new_momentum_array}


        return new_momentum_array

    def batched_single_node_momentum_update(self,initial_config, dt, start,end, staple_matrix_array): #ND numpy array nodecoords

        link_array = initial_config[0][start:end+1]
        momentum_array = initial_config[1][start:end+1]

        staple_matricies = staple_matrix_array[start:end+1]


        # indicies are node, direction mu, direction nu, first/second term, list of staple matricies

        # extract staple matricies
        firststaplematricies, secondstaplematricies = np.split(staple_matricies, 2, axis=3)

        staple11, staple12, staple13 = np.split(firststaplematricies, 3, axis=4)
        staple21, staple22, staple23 = np.split(secondstaplematricies, 3, axis=4)

        # calculate the twisted staple for each nu (index 2)
        firststaple = self.Barray[..., None, None][start:end+1] * staple11 @ staple12.conj().swapaxes(-1,
                                                                                                     -2) @ staple13.conj().swapaxes(
            -1, -2)
        secondstaple = self.V2_Barray[..., None, None][start:end+1] * staple21.conj().swapaxes(-1,
                                                                                              -2) @ staple22.conj().swapaxes(
            -1, -2) @ staple23

        # adding two terms together to get full staple then summing to get Vmu array. Squeeze to get rid of vestigal indicies
        staplesum = firststaple + secondstaple

        Varray = np.sum(staplesum, axis=2)
        Varray = np.squeeze(Varray)

        # calculating momentum update
        stapleterm = link_array @ Varray
        momentum_change = (1 / g ** 2) * (stapleterm.conj().swapaxes(-1, -2) - stapleterm) * dt

        new_momentum_array = momentum_array + momentum_change

        new_momentum_array = np.squeeze(new_momentum_array)
        # print(np.shape(new_momentum_array))

        # new_momentum={coordinates:new_momentum_array}

        return new_momentum_array


    def vectorized_momentum_update(self,initial_config,dt):

        link_dict = initial_config[0]
        momentum_array = np.array(list(initial_config[1].values()))
        link_array = np.array(list(link_dict.values()))
        staple_matricies = np.array([link_dict[link.node1.tuplecoords][link.direction] if link!=None else np.array([[0,0],[0,0]]) for link in self.staple_array.flatten()]).reshape(self.staple_array.shape + (2,2))

        #indicies are node, direction mu, direction nu, first/second term, list of staple matricies

        #extract staple matricies
        firststaplematricies, secondstaplematricies = np.split(staple_matricies, 2, axis = 3)
        staple11, staple12, staple13 = np.split(firststaplematricies, 3, axis = 4)
        staple21, staple22, staple23 = np.split(secondstaplematricies, 3, axis=4)

        # calculate the twisted staple for each nu (index 2)
        firststaple = self.Barray[...,None,None] * staple11 @ staple12.conj().swapaxes(-1,-2) @ staple13.conj().swapaxes(-1,-2)
        secondstaple = self.V2_Barray[...,None, None] * staple21.conj().swapaxes(-1,-2) @ staple22.conj().swapaxes(-1,-2) @ staple23


        #adding two terms together to get full staple then summing to get Vmu array. Squeeze to get rid of vestigal indicies
        staplesum = firststaple + secondstaple
        Varray = np.sum(staplesum, axis = 2)
        Varray = np.squeeze(Varray)

        #calculating momentum update
        stapleterm = link_array @ Varray
        momentum_change = (1/g**2) * (stapleterm.conj().swapaxes(-1,-2) - stapleterm) * dt

        #updating momentum
        new_momentum_array = momentum_array + momentum_change



        #print("vector", new_momentum_array)
        #new config
        new_config = [initial_config[0], dict(zip(initial_config[1].keys(), list(new_momentum_array)))]
        #print("elapsed", time.time()-starttime)
        return new_config



    def parallel_momentum_update(self, initial_config_arrays, dt):
        pool = self._pool
        link_array = initial_config_arrays[0]
        staple_matricies = np.array([link_array[link.node1.index_in_real_nodearray][link.direction] if link!=None else np.array([[0,0],[0,0]]) for link in self.staple_array.flatten()]).reshape(self.staple_array.shape + (2,2))


        inputs = [[initial_config_arrays, dt, node.index_in_real_nodearray, staple_matricies] for node in self.real_nodearray]

        new_momentum_array = np.stack(pool.starmap(self.single_node_momentum_update, inputs, chunksize = self.chunksize))
        return np.stack([link_array, new_momentum_array])






    def momentum_update(self, initial_config, dt):  # sends momentum at n*dt - dt/2 to n * dt + dt/2
        starttime = time.time()
        momentum_array = np.transpose(np.array(list(initial_config[1].values())), axes=[1, 0, 2,
                                                                                       3])  # array with dimensions [direction, node, matrix]
        momentum_dict = initial_config[1].copy()
        link_dict = initial_config[0]
        link_array = np.transpose(np.array(list(initial_config[0].values())), axes=[1,0,3,2])
        filled_config = self.ghost_fill_configuration(initial_config)

        filled_link_matricies_dict = filled_config[0]


        #link matricies dicts have indicies [direction, plane, node, matrix]

        Varray = np.empty((len(self.real_nodearray), self.dimensions), dtype = object)
        #do it slowly to make sure you're implementing EOM correctly. This conserves energy it's just slow as shit
        for nodeindex, node in enumerate(self.real_nodearray):
            for direction in range(self.dimensions):
                Vfirst = 0
                Vsecond = 0
                for sum_direction  in range(self.dimensions):
                    if sum_direction != direction:
                        Vfirst+=self.B(direction, sum_direction, node) * link_dict[self.translate(node, direction, 1).tuplecoords][sum_direction]\
                                @ link_dict[self.translate(node, sum_direction, 1).tuplecoords][direction].conj().T\
                                @ link_dict[node.tuplecoords][sum_direction].conj().T
                        Vsecond+=self.B(direction, sum_direction, self.translate(node, sum_direction, -1)) * link_dict[self.translate(self.translate(node,sum_direction, -1), direction, 1).tuplecoords][sum_direction].conj().T \
                                 @ link_dict[self.translate(node, sum_direction,-1).tuplecoords][direction].conj().T\
                                 @ link_dict[self.translate(node, sum_direction, -1).tuplecoords][sum_direction]

                Vdirection = Vfirst + Vsecond
                momentum_change = (1/g**2) * (Vdirection.conj().T @ link_dict[node.tuplecoords][direction].conj().T - link_dict[node.tuplecoords][direction] @ Vdirection)

                Varray[nodeindex, direction] = Vdirection
                momentum_array[direction, nodeindex] +=momentum_change*dt

        new_momentum_dict = dict(
            zip(initial_config[0].keys(), list(np.transpose(np.array(momentum_array), axes=[1, 0, 2, 3]))))

        #new_config = [initial_config[0], momentum_dict]
        #test:
        new_config = [initial_config[0], new_momentum_dict]


        #print("momentum gives:", momentum_array)

        #print("elapsed", time.time()-starttime)
        return new_config #works but is slow


    def evolution_step(self, config, dt):
        momentum_config = self.vectorized_momentum_update(config, dt)
        #momentum_config = self.momentum_update(config, dt)
        link_config = self.link_update(momentum_config, dt)
        #ham = self.hamiltonian(link_config)
        #hamiltonian_list.append(ham)
        #print("hamiltonian after step:", ham)
        return link_config

    def parallel_evolution_step(self, config, dt):
        pool = self._pool
        link_array = config[0]
        batch_size = int(np.ceil(self.num_nodes/self.processes))

        array_shape = np.shape(link_array)


        staple_matricies = np.array([link_array[link.node1.index_in_real_nodearray][
                                         link.direction] if link != None else np.array([[0, 0], [0, 0]]) for link in
                                     self.staple_array.flatten()]).reshape(self.staple_array.shape + (2, 2))

        momentum_inputs = [[config, dt, i * batch_size, np.min([(i + 1) * batch_size-1, self.num_nodes-1]), staple_matricies] for i in range(self.processes)]

        new_momentum_array = np.stack(pool.starmap(self.batched_single_node_momentum_update, momentum_inputs, chunksize=self.chunksize))

        new_momentum_array = np.reshape(new_momentum_array, newshape = array_shape)
        momentum_config = np.stack([link_array, new_momentum_array])

        link_inputs = [[momentum_config, dt,i * batch_size, np.min([(i + 1) * batch_size-1, self.num_nodes-1])] for i in range(self.processes)]

        new_link_array = np.stack(pool.starmap(self.batched_single_node_link_update, link_inputs, chunksize = self.chunksize))

        new_link_array = np.reshape(new_link_array, newshape=array_shape)
        return np.stack([new_link_array, new_momentum_array])

    def time_evolve(self, initial_config, evolution_time, nsteps=10000):
        starttime = time.time()
        config = initial_config
        dt = evolution_time / nsteps
        #ham = self.hamiltonian(config)
        #hamiltonian_list.append(ham)
        #print("initial Hamiltonian: ", ham)

        #config = self.momentum_update(config, dt/2)
        config = self.vectorized_momentum_update(config, dt / 2)

        config = self.link_update(config, dt)
        #print("starting main evolution")
        for i in range(nsteps-1):
            #print("Evolution step:", i)
            config = self.evolution_step(config, dt)

        #config = self.momentum_update(config, dt/2)
        config = self.vectorized_momentum_update(config, dt / 2)

        print("Elapsed time for time evolution:", time.time() - starttime)
        #ham = self.hamiltonian(config)
        #print("Final Hamiltonian", ham)
        #hamiltonian_list.append(ham)
        #plt.figure()
        #plt.plot(np.arange(0,nsteps+1, 1), hamiltonian_list, label = "energy")
        #plt.plot(np.arange(0, nsteps+1, 1), action_list, label = "action")
        #plt.plot(np.arange(0, nsteps+1, 1), momentum_list, label = "momentum")
        #plt.plot(np.arange(0, nsteps+1, 1)[1:], np.array(action_change_list)[1:]+np.array(momentum_change_list)[1:], label="changesum")
        #print(np.average(np.array(action_change_list)[1:]+np.array(momentum_change_list)[1:]) *nsteps)
        #plt.legend()
        return config



    def parallel_time_evolve(self, initial_config, evolution_time, nsteps=10000): #in this, configs are numpy arrays rather than dicts for performance reasons
        starttime = time.time()
        new_momentum_dict = {}
        new_link_dict = {}
        dt = evolution_time / nsteps

        link_dict = initial_config[0]
        momentum_dict = initial_config[1]

        link_array = np.array(list(link_dict.values()))
        momentum_array = np.array(list(momentum_dict.values()))

        config = np.stack([link_array, momentum_array])

        with mp.Pool(self.processes) as pool:
            self._pool = pool

            config = self.parallel_momentum_update(config, dt / 2)

            config = self.parallel_link_update(config, dt)
            for i in range(nsteps-1):
                config = self.parallel_evolution_step(config, dt)

            config = self.parallel_momentum_update(config, dt / 2)

        print("Elapsed time for time evolution:", time.time() - starttime)
        del self._pool
        for node in self.real_nodearray:
            new_momentum_dict[node.tuplecoords] = config[1][node.index_in_real_nodearray]
            new_link_dict[node.tuplecoords]=config[0][node.index_in_real_nodearray]
        config = [new_link_dict, new_momentum_dict]
        return config

    def generate_candidate_configuration(self, current_configuration, evol_time, number_steps):
        new_configuration = self.time_evolve(current_configuration, evol_time, nsteps = number_steps)
        return new_configuration

    def parallel_generate_candidate_configuration(self, current_configuration, evol_time, number_steps):
        new_configuration = self.parallel_time_evolve(current_configuration, evol_time, nsteps = number_steps)
        return new_configuration


    def hamiltonian(self, configuration): #action: sum(nodes in lattice) sum(mu < nu) Tr(1 - B_{mu nu}(n) U_{mu nu}(n))
        momentum_dict = configuration[1]
        momentum_array = np.array(list(momentum_dict.values()))

        filled_configuration = self.ghost_fill_configuration(configuration)
        filled_link_matricies_dict = filled_configuration[0]



        # calculating action of configuration
        #action = self.get_config_action(filled_configuration)
        runningsum = 0
        for node in self.real_nodearray:
            for direction1 in range(self.dimensions):
                for direction2 in range(self.dimensions):
                    if direction1 != direction2:
                        node1 = filled_link_matricies_dict[node.tuplecoords][direction1]
                        node2 = filled_link_matricies_dict[self.translate(node, direction1, 1).tuplecoords][direction2]
                        node3 = filled_link_matricies_dict[self.translate(node, direction2, 1).tuplecoords][direction1].conj().T
                        node4 = filled_link_matricies_dict[node.tuplecoords][direction2].conj().T
                        holonomy = node1 @ node2 @ node3 @ node4
                        runningsum += np.trace(np.matrix([[1,0],[0,1]]) - self.B(direction1, direction2, node) * holonomy)
        action = runningsum
        action = (1/g**2) * action

        # calculating fictious momentum contribution
        dagmomen = np.conj(np.transpose(momentum_array, axes = (0,1,3,2)))
        momentum_contrib = 0.5 * np.trace(np.sum(momentum_array @ dagmomen, axis=(0,1)), axis1 = -1, axis2 = -2)

        # returning Hamiltonian value
        #print("hailtonian function action:", action)
        hamiltonian = momentum_contrib + action
        if len(momentum_list)==0:
            momentum_change_list.append(momentum_contrib)
            action_change_list.append(action)
        else:
            momentum_change_list.append(momentum_contrib - momentum_list[-1])
            action_change_list.append(action - action_list[-1])

        momentum_list.append(momentum_contrib)
        action_list.append(action)
        #print("action:",action)
        #print("momentum contrib:", momentum_contrib)
        return np.real(hamiltonian)

    def get_config_action(self, configuration):


        filled_configuration = self.ghost_fill_configuration(configuration)

        filled_link_matricies_dict = filled_configuration[0]

        # calculating action of configuration
        node_action_contribs = 0
        for plane in self.planeslist:  # generate all plaquette holonomies
            node_plaquette_corners = []
            Blist = []
            for node in self.real_nodearray:
                node_plaquette_corners.append(self.get_plaquette_corners(node, plane))
                Blist.append(self.B(plane[0], plane[1], node))
            node_plaquette_corners = np.array(node_plaquette_corners).T
            Blist = np.array(Blist)
            first_link_matricies = np.array(
                [link_matricies[plane[0]] for link_matricies in
                 [filled_link_matricies_dict[node.tuplecoords] for node in node_plaquette_corners[0]]])
            second_link_matricies = np.array(
                [link_matricies[plane[1]] for link_matricies in
                 [filled_link_matricies_dict[node.tuplecoords] for node in node_plaquette_corners[1]]])
            third_link_matricies = np.array(
                [link_matricies[plane[0]].conj().T for link_matricies in
                 [filled_link_matricies_dict[node.tuplecoords] for node in node_plaquette_corners[3]]])
            fourth_link_matricies = np.array(
                [link_matricies[plane[1]].conj().T for link_matricies in
                 [filled_link_matricies_dict[node.tuplecoords] for node in node_plaquette_corners[0]]])
            # plane_holonomies: gives array of each matrix corresponding to the holonomy around a plaquette
            # in list corresponding to posiiton of corner node in self.real_nodearray
            plane_holonomies = Blist[:, np.newaxis, np.newaxis] * (
                (first_link_matricies @ second_link_matricies @ third_link_matricies @ fourth_link_matricies))
            node_action_contribs += np.sum(np.trace(plane_holonomies, axis1=1, axis2=2))

            node_action_contribs = node_action_contribs
        action =  (2/g**2) * (len(self.planeslist) * self.num_nodes * 2 - node_action_contribs)
        return action

    def accept_config(self, new_configuration, initial_configuration, observables = None):
        observablelist = []
        #print("initial Hamiltonian, Action")
        Hinitial = self.hamiltonian(initial_configuration)
        #Ainitial = self.get_config_action(initial_configuration)
        #print(Hinitial, Ainitial)

        #print("action:", Ainitial)
        #print("final Hamiltonian")
        Hnew = self.hamiltonian(new_configuration)
        #print(Hnew)
        difference = Hnew - Hinitial
        transition_prob = np.minimum(1, np.exp(-difference))
        randomvar = random.uniform(0, 1)
        print("Hamiltonian difference:", difference, "Configuration transition probability:", transition_prob)
        #print("Hamiltonian difference over initial Hamiltonian:", np.abs(difference/Hinitial))
        if randomvar < transition_prob:
            observable_dict = {}
            new_matrix_dict = dict(zip(self.real_nodearray, new_configuration[0].values()))
            for node in self.real_nodearray:
                for direction in range(self.dimensions):
                    self.link_dict[node][direction].set_matrix(new_matrix_dict[node][direction])
            if observables!= None:
                for observable in observables:
                    observable_dict[observable.identifier] = tuple(observable.evaluate(self))
                return True, observable_dict
            else:
                return True, None
        else:
            return False, None

    def chain(self, number_iterations, evolution_time, number_steps, observables = None, filename = None, log = False):
        """global hamiltonian_list
        global action_list
        global momentum_list
        global action_change_list
        global momentum_change_list"""
        starttime = time.time()
        observable_list = []
        acceptances = 0
        momentum = self.random_momentum()
        for i in range(number_iterations):
            print("Iteration:", i)
            hamiltonian_list = []
            action_list = []
            momentum_list = []
            action_change_list = []
            momentum_change_list = []
            holder = []
            for arraylist in list(momentum.values()):
                newlist = []
                for array in arraylist:
                    newlist.append(array.copy())
                holder.append(newlist)
            #print(holder[0])
            old_config = [self.get_link_matrix_dict(), momentum.copy()]
            candidate = self.generate_candidate_configuration(old_config, evolution_time, number_steps)
            #print(holder[0])
            old_config = [self.get_link_matrix_dict(), dict(zip(momentum.keys(),holder))]
            accepted, data = self.accept_config(candidate, old_config, observables = observables)
            if accepted==True:
                observable_list.append(data)
                print("accepted")
                acceptances+=1
            momentum = self.random_momentum()
        print("Avg time per config: ", (time.time()-starttime)/number_iterations)
        print("acceptance rate:", acceptances/number_iterations)

        hamiltonian_list = []
        action_list = []
        momentum_list = []
        momentum_change_list = []
        action_change_list = []
        #save the measured observables
        if (log == True or filename!=None):
            if filename!=None:
                with open(filename + str([obs.identifier for obs in observables]) + str(self.shape) + "_twists:" + str(self.twists) + ".txt", "w") as f:
                    for item in observable_list:
                        f.write(f"{item}\n")
            else:
                with open(str(time.strftime("%d-%m-%Y_%H:%M:%S")) + str([obs.identifier for obs in observables]) + str(self.shape) + "_twists:" + str(self.twists) +".txt", "w") as f:
                    for item in observable_list:
                        f.write(f"{item}\n")
        return observable_list


    def parallel_chain(self, number_iterations, evolution_time, number_steps, observables = None, filename = None, log = False):
        starttime = time.time()
        observable_list = []
        acceptances = 0
        momentum = self.random_momentum()
        for i in range(number_iterations):
            print("Iteration:", i)
            holder = []
            for arraylist in list(momentum.values()):
                newlist = []
                for array in arraylist:
                    newlist.append(array.copy())
                holder.append(newlist)
            old_config = [self.get_link_matrix_dict(), momentum.copy()]
            candidate = self.parallel_generate_candidate_configuration(old_config, evolution_time, number_steps)
            old_config = [self.get_link_matrix_dict(), dict(zip(momentum.keys(),holder))]
            accepted, data = self.accept_config(candidate, old_config, observables = observables)
            if accepted==True:
                observable_list.append(data)
                print("accepted")
                acceptances+=1
            momentum = self.random_momentum()
        print("Avg time per config: ", (time.time()-starttime)/number_iterations)
        print("acceptance rate:", acceptances/number_iterations)

        #save the measured observables
        if (log == True or filename!=None):
            if filename!=None:
                with open(filename + str([obs.identifier for obs in observables]) + str(self.shape) + "_twists:" + str(self.twists) + ".txt", "w") as f:
                    for item in observable_list:
                        f.write(f"{item}\n")
            else:
                with open(str(time.strftime("%d-%m-%Y_%H:%M:%S")) + str([obs.identifier for obs in observables]) + str(self.shape) + "_twists:" + str(self.twists) +".txt", "w") as f:
                    for item in observable_list:
                        f.write(f"{item}\n")
        return observable_list







