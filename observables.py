
#observables for 1^4 lattice
def plaquette_observable(lattice):
    outputlist = []
    node = lattice[0,0,0,0]
    for plane in lattice.planeslist:
        matrix_holonomy = lattice.get_plaquette_holonomy([0,0,0,0],plane)
        plaquette = lattice.B(plane[0], plane[1], node) * np.trace(matrix_holonomy)
        readable_plane = (plane[0]+1, plane[1]+1)
        outputlist.append([readable_plane, plaquette])
    return outputlist

def winding_loop(lattice):
    outputlist = []
    node = lattice[0,0,0,0]
    for direction in range(lattice.dimensions):
        outputlist.append([direction+1, np.trace(node.get_link(direction,0).get_matrix())])
    return outputlist