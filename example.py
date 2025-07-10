import numpy as np
from search.naive import Naive
from search.kd_tree import KDTree


vector = np.array([[0., 0., 1.],
                    [1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.],
                    [4., 4., 4.],
                    [5., 5., 5.],
                    [6., 6., 6.],
                    [7., 7., 7.],
                    [8., 8., 8.],
                    [9., 9., 9.]])

kdtree = KDTree(vector, 3)

node, score = kdtree.search(vector[0])
print(kdtree.root)
bol = vector[0] == vector[node.idx]