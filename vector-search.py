import heapq
import numpy as np

from distance import cos_similarity

class Naive:

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.store = []

    def add(self, vector: np.ndarray):
        self.store.append(vector)

    def search(self, query: np.ndarray, k: int = 1):
        distances = []
        result = []

        for i, vec in enumerate(self.store):
            dist = cos_similarity(query, vec)
            heapq.heappush(distances, (dist, i))

        for i in range(k):
            best = heapq.heappop(distances)
            result.append(best)

        return result

            

            
            

            
    
    
        

