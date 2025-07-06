

import numpy as np
from collections import deque

class KDNode:
    def __init__(self, idx: int, dimension: int, range: list[int], left: 'KDNode' | None = None, right: 'KDNode' | None = None, parent: 'KDNode' | None = None):
        self.idx = idx  # look up index to find vector
        self.dimension = dimension  # current dimension
        self.left = left  # left subtree
        self.right = right  # right subtree
        self.range = range
        self.parent = parent


class KDTree:

    def __init__(self, vectors: np.ndarray, dimensions):
        # TODO check dimensions
        self.dimensions = dimensions
        self.store = vectors
        self.root = self._build_tree(vectors)


    def search(self, query_vector: np.ndarray):
        stack: list[KDNode] = []

        stack.append(self.root)

        node: KDNode | None = None

        # traverse down
        while stack:
            node = stack.pop()
            
            if node.left and self.store[node.left.idx, node.dimensions] <= query_vector[node.dimension]:
                stack.append(node.left)
            
            if node.right and self.store[node.left.idx, node.dimensions] < query_vector[node.dimension]:
                stack.append(node.right)

        # TODO backtrack and prune

        
    # TODO Node based approach adds extra memory overhead when creating object
    # Look into array based or heap approach
    def _build_tree(self, vectors: np.ndarray) -> KDNode:

        # sort vectors by current dimension
        sorted_indices = vectors[:, 0].argsort()
        median_idx = vectors[(0 + len(vectors)) / 2]
        root = KDNode(median_idx, 0, [0, len(self.store)])
        stack: list[KDNode] = []

        while stack:
            node = stack.pop()
            next_dimension = (node.dimension + 1) % self.dimensions

            # TODO Can we do this without creating a new array
            # TODO Maybe should cache sorted vectors

            # sort vectors by current dimension
            sorted_indices = vectors[:, next_dimension].argsort()
            
            start = node.range[0]
            end = node.range[1]

            if start == end:
                continue

            left_range = [start, node.idx]
            right_range = [node.idx, end]

            right_median_idx = sorted_indices[(right_range[0] + right_range[1]) / 2]
            left_median_idx = sorted_indices[(left_range[0] + left_range[1]) / 2]

            # add right node
            right_child = KDNode(right_median_idx, next_dimension, right_range, node)
            node.right = right_child
            stack.append(right_child)

            # add left node to stack
            left_child = KDNode(left_median_idx, next_dimension, left_range, node)
            node.left = left_child
            stack.append(left_child)
        
        return root
    

         
                