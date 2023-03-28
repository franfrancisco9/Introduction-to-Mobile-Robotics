# Determin whether a matrix is orthogonal

import numpy as np

def is_orthogonal(A):
    # It checks if the matrix is orthogonal
    # A: matrix to check
    # return: True if orthogonal, False otherwise
    return np.allclose(np.dot(A, A.T), np.eye(A.shape[0]))


if __name__ == "__main__":
    A = 1/3 * np.array([[2, 2, -1], [2, -1, 2], [-1, 2, 2]])
    print("A is orthogonal: ", is_orthogonal(A))
    