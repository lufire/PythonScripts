# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 17:34:06 2019

@author: lukas
"""

import numpy as np
import scipy as sp
from scipy.interpolate import griddata, interp1d

def calc_nodes_1d(elements):
    """
    Calculates an node 1-d-array from an element 1-d-array,
    uses the [:, 1], [:, -2] entries of the calculated node 1-d-array
    to fill the first als last row of the node 1-d-array.
    """
    nodes = np.asarray((elements[:-1] + elements[1:])) * .5
    return np.hstack([elements[0], nodes, elements[-1]])#

def calc_elements_1d(nodes):
    """
    Calculates an node 1-d-array from an element 1-d-array,
    uses the [:, 1], [:, -2] entries of the calculated node 1-d-array
    to fill the first als last row of the node 1-d-array.
    """
    return np.asarray((nodes[:-1] + nodes[1:])) * .5

n_ele = 100
n_nodes = n_ele + 1

l = 1.0
nodes = np.linspace(0.0, l, n_nodes)
elements = calc_elements_1d(nodes)

uniform_element_coord = elements
uniform_node_coord = nodes

def interpolate_to_nodes(element_field, elements=uniform_element_coord, 
                         nodes=uniform_node_coord, kind='linear'):
    f = interp1d(elements, element_field, kind, 
                 fill_value=(element_field[0], element_field[-1]))
    return f(nodes)


def interpolate_to_elements(node_field, nodes=uniform_node_coord,
                            elements=uniform_element_coord,
                            kind='linear'):
    return interp1d(nodes, node_field, kind)




a = np.random.rand(n_nodes)

print(a)
print(np.diff(a))



print(interpolate_to_nodes(np.ediff1d(a)))
print(calc_nodes_1d(np.ediff1d(a)))

