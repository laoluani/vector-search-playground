import heapq
import numpy as np

from distance import cos_similarity

class Naive:

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.store = []

    def add(self, vector: np.ndarray):
        self.store.append(vector)

    def search(self, query: np.ndarray, k: int = 1) -> list[tuple[float, np.ndarray]]:
        distances = []
        result = []

        for i, vec in enumerate(self.store):
            dist = cos_similarity(query, vec)
            # make max heap by negating distance
            heapq.heappush(distances, (-dist, i))

        for i in range(k):
            best_dist, best_index = heapq.heappop(distances)
            # negate distance to return to orignal value
            result.append((-best_dist, self.store[best_index]))

        return result

            

            
            

            
    
    
        

