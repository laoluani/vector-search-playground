import numpy as np
from search.naive import Naive
from search.kd_tree import KDTree


vector = np.random.rand(10, 3)
kdtree = KDTree(vector, 3)

node, score = kdtree.search(vector[0])
print(kdtree.root)
bol = vector[0] == vector[node.idx]