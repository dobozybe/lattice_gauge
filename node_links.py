import numpy as np

class Node:
    def __init__(self, coordinates, is_ghost_node = False):
        self.coordinates = coordinates
        self.links = []
        self.ghost_node = is_ghost_node
        self.dimensions = len(coordinates)
        for direction in range(len(coordinates)):
            self.links.append([None, None])

    def addlink(self, link, orientation):
        self.links[link.direction][orientation] = link  # orientation 0 means node is connected to node w/greater index (1 means smaller)
        return

    def getlinks(self):
        return self.links

    def get_link(self, direction, orientation):
        return self.links[direction][orientation]

    def get_next_node(self,direction, orientation):
        #gets next node, orientation 0 is next node in positive direction and orientation 1 is backwards.
        thoselinks = self.links[direction]
        directed_link = thoselinks[orientation]
        if orientation == 0:
            return directed_link.node2
        else:
            return directed_link.node1
    def __str__(self):
        return str(self.coordinates)




class Link:  # links are oriented from node1 to node2
    def __init__(self, node1, node2, direction):
        # node1 and node2 are node objects
        # direction is an integer that should correspond to lattice axes
        self.node1 = node1
        self.node2 = node2
        self.direction = direction
        self.student_link = None
        self.matrix = np.diag(np.full(2, 1))
        node1.addlink(self, 0)
        node2.addlink(self, 1)

    def get_matrix(self): #if link connects node to node further along the lattice, return matrix. If it points back, return conjugate of matrix
        return self.matrix


    def set_matrix(self, newmatrix):
        self.matrix = newmatrix
        if self.student_link != None:
            self.student_link.update()

    def set_student_link(self,link):
        self.student_link = link

    def __str__(self):
        if self.node1.coordinates[self.direction] < self.node2.coordinates[self.direction]:
            return str(self.node1)+ "--->"+ str(self.node2)
        else:
            return str(self.node2) + "<---" + str(self.node1)


class StudentLink(Link):
    def __init__(self, node1, node2, direction):
        # node1 and node2 are node objects
        # direction is an integer that should correspond to lattice axes
        self.node1 = node1
        self.node2 = node2
        self.direction = direction
        self.parent_link = None
        self.student_link = None #should always be none. Students shouldn't have students.
        self.matrix = np.diag(np.full(2, 1))
        node1.addlink(self, 0)
        node2.addlink(self, 1)

    def set_parent(self,link):
        self.parent_link = link
        self.parent_link.set_student_link(self)

    def update(self):
        new_matrix = self.parent_link.get_matrix() ##todo make work for twists
        self.set_matrix(new_matrix)