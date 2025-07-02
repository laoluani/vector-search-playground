import heapq
import numpy as np

from distance import cos_similarity

class Naive:

    def __init__(self, dimension):
        self.dimension = dimension
        self.store = []

    def add(self, vector):
        self.store.append(vector)

    def search(self, query, k = 1):
        distances = []
        result = []

        for i, vec in enumerate(self.store):
            dist = cos_similarity(query, vec)
            heapq.heappush(distances, (dist, i))

        for i in k:
            best = heapq.heappop(distances)
            result.append(best)

        return result

            

            
            

            
    
    
        

