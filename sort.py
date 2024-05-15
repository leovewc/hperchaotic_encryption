import numpy as np
import cv2
import hashlib
import matplotlib.pyplot as plt

def sort_chaotic_sequence(X1, M, N):

    X1_matrix = X1.reshape((M, N))

    # Initial
    I1 = np.zeros_like(X1_matrix)
    X1_prime = np.zeros_like(X1_matrix, dtype=int)

    for i in range(M):
        indices = np.argsort(X1_matrix[i, :])
        I1[i, :] = X1_matrix[i, indices]
        X1_prime[i, :] = indices

    return I1, X1_prime
def permute_and_sort_sequences(X, X_prime, M, N):

    X_matrix = X.reshape((M, N))
    permuted_matrix = np.zeros_like(X_matrix)
    sorted_matrix = np.zeros_like(X_matrix)
    new_indices = np.zeros_like(X_matrix, dtype=int)

    for i in range(M):
        permuted_matrix[i, :] = X_matrix[i, X_prime[i, :]]

    for i in range(M):
        indices = np.argsort(permuted_matrix[i, :])
        sorted_matrix[i, :] = permuted_matrix[i, indices]
        new_indices[i, :] = indices

    return sorted_matrix, new_indices
