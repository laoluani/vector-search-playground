import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from utils.distance import cos_similarity

class ProductQuantisation:
    """Product Quantisation class."""
    n_subvectors = 8 # Num subvectors to split vector in for PQ
    n_clusters = 256 # Num clusters for each subspace of subvectors

    def __init__(self, dimension: int):

        self.dimension = dimension
        self.store = []
        self.centroids = []
        self.is_trained = False # False if indexing hasn't been run else True

    def add(self, vector: np.ndarray):
        """Add vector(s) to the store and assign index."""
        if not self.is_trained:
            raise RuntimeError("PQ must be trained before adding new vectors.")
        
        if vector.shape[-1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {vector.shape[0]}.")
        
        norm_vector = normalize(vector.reshape(1, -1), axis=1)[0]
        subvector = norm_vector.reshape(1, ProductQuantisation.n_subvectors, -1).transpose(1, 0 , 2)
        

        distances = cos_similarity(self.centroids, norm_vector)
        closest_centroid_id = np.argmin(distances)

        # Add the vector to the store and update the centroid_index
        new_vector_index = len(self.store)
        self.store.append(vector)
        self.centroid_index[closest_centroid_id].append(new_vector_index)

    def train(self, vectors: np.ndarray):
        """Product quantise the vectors."""
        if vectors.shape[-1] != self.dimension:
            raise ValueError(f"Training vectors dimension mismatch. Expected {self.dimension}, got {vectors.shape[1]}.")

        n_vectors = len(vectors)
        # Subvectors of shape (n_subvectors, n_vectors, dimension)
        subvectors = vectors.reshape(n_vectors, ProductQuantisation.n_subvectors, -1).transpose(1, 0 , 2)

        # Initialise the product quantised vectors as array of 0s
        pq_vectors = np.zeros((ProductQuantisation.n_subvectors, len(vectors)), dtype=np.uint8)

        for n, vec in enumerate(subvectors):
            norm_vec = normalize(vec, axis=1) # Normalise to ensure Euclidean distance is proportional to Cosine sim
            kmeans = KMeans(n_clusters=ProductQuantisation.n_clusters, random_state=42)
            kmeans.fit(norm_vec)

            self.centroids.append(normalize(kmeans.cluster_centers_, axis=1))
            pq_vectors[n] = kmeans.predict(norm_vec) # Predicts for each subvector which centroid it's closest to

        self.store = pq_vectors.T.astype(np.uint8) # Transpose to be shape (len(vectors), num_subvectors)
        self.is_trained = True

    def search(self, query_vector: np.ndarray, k: int = 1):
        """Find top k most similar vectors in store."""
        norm_query_vector = normalize(query_vector.reshape(1, -1), axis=1)[0]
        query_subvectors = norm_query_vector.reshape(ProductQuantisation.n_subvectors, -1)

        centroid_distances = {}
        for n, query_subvec in enumerate(query_subvectors):
            centroid_distances[n] = cos_similarity(self.centroids[n], query_subvec)

        scores = []
        for vec in self.store:  # vec is a list of centroid indices, length = num_subvectors
            score = sum(centroid_distances[n][int(idx)] for n, idx in enumerate(vec))
            scores.append(score)
        return np.argsort(scores)[::-1][:k] 