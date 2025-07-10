from __future__ import annotations

import numpy as np

from collections import deque
from dataclasses import dataclass
from utils.distance import cos_similarity


@dataclass
class KDNode:
    idx: int
    dimension: int
    range: list[int]
    left: KDNode | None = None
    right: KDNode | None = None
    parent: KDNode | None = None
    is_left: bool = True


class KDTree:

    def __init__(self, vectors: np.ndarray, dimensions):
        # TODO check dimensions
        self.dimensions = dimensions
        self.store = vectors
        self.root = self._build_tree(vectors)
        self.size = 0


    def search(self, query_vector: np.ndarray):
        stack: list[KDNode] = []

        stack.append(self.root)

        best_node: KDNode | None = None

        # traverse down
        while stack:
            node = stack.pop()
            
            if node.left and query_vector[node.dimension] <= self.store[node.idx, node.dimension]:
                stack.append(node.left)
            
            if node.right and query_vector[node.dimension] > self.store[node.idx, node.dimension]:
                stack.append(node.right)

            best_node = node

        best_distance = cos_similarity(self.store[best_node.idx], query_vector)
        
        stack.append(best_node.parent)

        while stack:
            node = stack.pop()
            distance = cos_similarity(self.store[node.idx], query_vector)
            
            if distance > best_distance:
                best_node = node
                best_distance = distance
                # explore sibling node
                n = node.right if node.is_left else node.left
                if n:
                    stack.append(n)
            else:
                if node.parent:
                    stack.append(node.parent)

        return best_node, best_distance

        
    # TODO Node based approach adds extra memory overhead when creating object
    # Look into array based or heap approach
    def _build_tree(self, vectors: np.ndarray) -> KDNode:

        # sort vectors by current dimension
        vector_idx = list(range(len(vectors)))
        vector_idx.sort(key=lambda idx: vectors[idx][0])
        start = 0 
        end = len(vectors) - 1

        median_idx = vector_idx[int(end / 2)]
        root = KDNode(median_idx, 0, [0, end])

        queue = deque()
        queue.append(root)


        while queue:
            node = queue.popleft()

            start = node.range[0]
            end = node.range[1]

            if start == end:
                continue

            next_dimension = (node.dimension + 1) % self.dimensions

            # TODO Can we do this without creating a new array
            # TODO Maybe should cache sorted vectors

             # sort vectors by current dimension
            idx_slice = vector_idx[start:end + 1]
            idx_slice.sort(key=lambda idx: vectors[idx][next_dimension])
            vector_idx[start:end + 1] = idx_slice
            
            # TODO This should be the median range
            mid = int((start + end) / 2)
            left_range = [start, mid]
            right_range = [mid + 1, end]

            right_median_idx = vector_idx[int((right_range[0] + right_range[1]) / 2)]
            left_median_idx = vector_idx[int((left_range[0] + left_range[1]) / 2)]

            # add left node to stack
            left_child = KDNode(left_median_idx, next_dimension, left_range, None, None, node, True)
            node.left = left_child
            left_child.parent = node
            queue.append(left_child)

            # add right node
            right_child = KDNode(right_median_idx, next_dimension, right_range, None, None, node, False)
            node.right = right_child
            right_child.parent = node
            queue.append(right_child)
        
        return root
    

         
                