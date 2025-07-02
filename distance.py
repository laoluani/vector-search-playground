import numpy as np

    # Example vectors
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

def cos_similarity(a, b):
    dot_product = np.dot(A, B)
    magnitude_A = np.linalg.norm(A)
    magnitude_B = np.linalg.norm(B)

    return dot_product / (magnitude_A * magnitude_B)