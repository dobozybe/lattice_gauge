import numpy as np
import matplotlib.pyplot as plt
from sympy.utilities.iterables import multiset_permutations
from sympy.functions.special.tensor_functions import eval_levicivita
import time
import utilities

#identifiers:
#One_Cube_Plaquette: onecubeplaquette
#One_Cube_WInding: onecubewinding


def handle_observables(filename, observables):
    data_dict = {}
    with open(filename +".txt", "r") as f:
        loaded_list = f.read().splitlines()
    for index, observable in enumerate(observables):
        observablelist = []
        for entry in loaded_list:
            observablelist.append(eval(entry, {"np": np, "I":1j})[observable.identifier])
        data_dict[observable.identifier] = observablelist
    return data_dict


def plot_hist(data, title):
    plt.figure(figsize=(12, 6))  # Make figure wider
    datacounts, dataedges = np.histogram(data, bins=50)
    plt.bar(dataedges[:-1], datacounts, width=np.diff(dataedges), edgecolor='none')
    plt.title(title)
    plt.tight_layout()

class Observable:
    def __init__(self):
        self.identifier = None
    def evaluate(self, lattice):
        pass
    def visualize(self, data):
        pass



#observables for 1^4 lattice
class One_Cube_Plaquette(Observable):
    def __init__(self):
        self.identifier = "onecubeplaquette"
    def evaluate(self, lattice):
        outputlist = []
        #node = lattice[0, 0, 0, 0]
        node = lattice.real_nodearray[0]
        for plane in lattice.planeslist:
            matrix_holonomy = lattice.get_plaquette_holonomy(node.coordinates, plane)
            #plaquette = lattice.B(plane[0], plane[1], node) * np.trace(matrix_holonomy)
            plaquette = np.trace(matrix_holonomy)
            readable_plane = (plane[0] + 1, plane[1] + 1)
            outputlist.append([readable_plane, plaquette])
        return outputlist

    def visualize(self, data):
        datalist = data[self.identifier]
        measurementarray = np.array([[datapoint[i][1] for datapoint in datalist] for i in range(len(datalist[0]))])
        planearray = np.array([[datapoint[i][0] for datapoint in datalist] for i in range(len(datalist[0]))])
        for i in range(len(measurementarray)):
            plot_hist(measurementarray[i], "Value of " + str(planearray[i][0]) + " plaquette")


class One_Cube_Winding(Observable):
    def __init__(self):
        self.identifier = "onecubewinding"
    def evaluate(self, lattice):
        outputlist = []
        node = lattice[0, 0, 0, 0]
        for direction in range(lattice.dimensions):
            outputlist.append([direction + 1, np.trace(node.get_link(direction, 0).get_matrix())])
        return outputlist
    def visualize(self, data):
        datalist = data[self.identifier]
        onedata = []
        twodata = []
        threedata = []
        fourdata = []
        for list in datalist:
            onedata.append(list[0][1])
            twodata.append(list[1][1])
            threedata.append(list[2][1])
            fourdata.append(list[3][1])
        plot_hist(onedata, "U1 Trace")
        plot_hist(twodata, "U2 Trace")
        plot_hist(threedata, "U3 Trace")
        plot_hist(fourdata, "U4 Trace")

class General_Winding(Observable):
    def __init__(self, basenodecoords):
        self.basenodecoords = basenodecoords
        self.identifier = "genwinding" + str(self.basenodecoords)
    def evaluate(self, lattice):
        start = time.time()
        shape = lattice.shape
        productarray = np.full((lattice.dimensions, 2,2), fill_value=np.array([[1,0],[0,1]]), dtype = np.complex128)
        outputlist = []
        linkdicts = lattice.get_link_matrix_dict()
        for direction in range(lattice.dimensions):
            currentnode = lattice[self.basenodecoords]
            for i in range(shape[direction]):
                productarray[direction] = linkdicts[currentnode.tuplecoords][direction] @ productarray[direction]
                currentnode = lattice.translate(currentnode, direction, 1)
        holonomyarray = 0.5 * np.trace(productarray, axis1= 1, axis2 = 2)
        for direction in range(lattice.dimensions):
            outputlist.append([direction, holonomyarray[direction]])
        #print("genwinding eval", time.time()-start)
        return outputlist
    def visualize(self, data):
        datalist = data[self.identifier]
        holonomies_array = np.array([[datalist[config][direction][1] for config in range(len(datalist))] for direction in range(np.shape(datalist)[1])])
        for direction in range(len(holonomies_array)):
            plot_hist(holonomies_array[direction], "Winding loop in direction " + str(direction+1) + "with basepoint: " + str(self.basenodecoords))

class Action(Observable):
    def __init__(self, deformation_coeff = 0):
        self.identifier = "action"
        self.deformation_coeff = deformation_coeff
    def evaluate(self, lattice):
        start = time.time()
        links = lattice.get_link_matrix_dict()
        momentum = dict(zip(links.keys(),np.zeros(np.shape(list(links.values())))))
        config = [links,momentum]

        if self.deformation_coeff !=0:
            deformation_list = [1]  # should be 0/1 for yes/no, index_incrememnt, time_length
            index_increment = 1
            for i in range(1, len(lattice.shape)):
                index_increment *= lattice.shape[i]
            deformation_list.append(index_increment)
            deformation_list.append(lattice.shape[0])
        deformation_data = [1, [deformation_list, self.deformation_coeff]]

        datalist = [lattice.get_config_action(config, deformation_data=deformation_data)[0]]
        #print("action eval", time.time()-start)
        print("action after time evolution", datalist[0])
        return datalist
    def visualize(self, data):
        datalist = data[self.identifier]

        plot_hist(datalist, "Action Histogram")


class TopologicalCharge(Observable):
    def __init__(self):
        self.identifier = "topcharge"
    def evaluate(self, lattice):
        start = time.time()
        link_array = np.array(list(lattice.get_link_matrix_dict().values()))

        momentum_array = np.array([])

        plaquette_matricies = utilities.make_plaquette_array([link_array, momentum_array],
                                                             lattice.plaquette_corner_index_array)
        first_link, second_link, third_link, fourth_link = np.split(plaquette_matricies, 4, axis=3)

        plaquette_holonomies = first_link @ second_link @ third_link.conj().transpose(0, 1, 2, 3, 5,
                                                                                      4) @ fourth_link.conj().transpose(
            0, 1, 2, 3, 5, 4)

        perms = np.array(list(multiset_permutations(np.array(range(lattice.dimensions)))))
        levi_civita_values = np.array([eval_levicivita(*perm) for perm in perms])
        print(np.shape(levi_civita_values))

        plaq1 = plaquette_holonomies[:, perms[:, 0], perms[:, 1]]
        plaq2 = plaquette_holonomies[:, perms[:, 2], perms[:, 3]]

        products = plaq1 @ plaq2

        traces = np.trace(products, axis1= -1, axis2=-2)
        traces = traces.squeeze()


        summed_traces = np.sum(traces, axis = 0)
        perm_traces = levi_civita_values * summed_traces



        runningsum = np.sum(perm_traces)
        datalist = [(-1/(32 * np.pi**2)) * runningsum]

        return datalist




    def visualize(self, data):
        datalist = data[self.identifier]
        plot_hist(datalist, "Topological Charge Histogram")


