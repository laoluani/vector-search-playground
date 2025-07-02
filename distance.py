import numpy as np

    # Example vectors
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

def cos_similarity(a: np.ndarray, b: np.ndarray):
    return  np.dot(a, b) / (np.linalg.norm(a) *  np.linalg.norm(b))