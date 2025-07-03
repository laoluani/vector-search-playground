import heapq
import numpy as np

from distance import cos_similarity

class Naive:
    """A naive brute force vector search implementation using cosine similarity."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.store = []

    def add(self, vector: np.ndarray):
        """Adds vector(s) to the store."""
        assert vector.shape[-1] == self.dimension, f"Vector(s) must match the specified dimension {self.dimension}."

        if vector.ndim != 1:  # Add batch of vectors if not a single vector
            for vec in vector:
                self.store.append(vec)
        else:
            self.store.append(vector)

    def search(self, query_vector: np.ndarray, k: int = 1) -> list[tuple[float, np.ndarray]]:
        """Searches for the k most similar vectors to the query vector using cosine similarity."""
        distances = []
        result = []

        for i, vec in enumerate(self.store):
            dist = cos_similarity(query_vector, vec)
            # make max heap by negating distance
            heapq.heappush(distances, (-dist, i))

        for i in range(k):
            best_dist, best_index = heapq.heappop(distances)
            # negate distance to return to orignal value
            result.append((-best_dist, self.store[best_index]))

        return result

            

            
            

            
    
    
        

