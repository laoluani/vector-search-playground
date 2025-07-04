"""Inverted File Indexing"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from utils.distance import cos_similarity

class IVF:
    """Inverted File Indexing class"""

    def __init__(self, dimension: int):
        
        self.dimension = dimension
        self.store = []
        self.centroid_index: dict | None = None
        self.centroids: np.ndarray | None = None
        self.is_trained = False # False if indexing hasn't been run else True

    def add(self, vector: np.ndarray):
        """Add vector(s) to the store and assign index."""
        if not self.is_trained:
            raise RuntimeError("IVF must be trained before adding new vectors.")

        if vector.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {vector.shape[0]}.")
        if vector.ndim!=1:
            for vec in vector:
                distances = cos_similarity(self.centroids, vec)
                closest_centroid_id = np.argmin(distances)

                new_vector_index = len(self.store)
                self.store.append(vector)
                self.centroid_index[closest_centroid_id].append(new_vector_index)
        else:
            distances = cos_similarity(self.centroids, vector)
            closest_centroid_id = np.argmin(distances)

            # Add the vector to the store and update the centroid_index
            new_vector_index = len(self.store)
            self.store.append(vector)
            self.centroid_index[closest_centroid_id].append(new_vector_index)

    def search(self, query_vector: np.ndarray, k: int = 1):
        """Search for top k most similar vectors."""
        centroid_distances = cos_similarity(self.centroids, query_vector)
        closest_centroid = np.argmin(centroid_distances)

        vector_indices = self.centroid_index.get(closest_centroid)
        vectors = np.array([self.store[idx] for idx in vector_indices])
        
        nn = NearestNeighbors(n_neighbors=min(k, len(vectors)), algorithm='brute', metric='cosine') # Search for vector(s) closest
        nn.fit(vectors)
        
        distances, indices = nn.kneighbors(query_vector.reshape(1, -1))

        # Map back to original indices in self.store
        results = [(dist, self.store[vector_indices[idx]]) for idx, dist in zip(indices[0], distances[0])]
        return results

    def train(self, vectors: np.ndarray, n_clusters: int = 10):
        """Cluster vectors with KMeans and determine centroid and centroid indices."""
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Training vectors dimension mismatch. Expected {self.dimension}, got {vectors.shape[1]}.")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(vectors)
        self.centroids = kmeans.cluster_centers_

        self.centroid_index = {i: [] for i in range(n_clusters)}
        for i, label in enumerate(kmeans.labels_):
            self.centroid_index[label].append(i)
        
        self.store = vectors.tolist()
        self.is_trained = True


