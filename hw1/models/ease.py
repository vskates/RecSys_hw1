"""EASE: замкнутое решение с нулевой диагональю."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class EASE:
    def __init__(self, regularization=100.):
        self.regularization = regularization

    def fit(self, history, shape):
        from utils import matrix
        self.history = matrix(history, shape, positive_only=True)
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::shape[1] + 1] += self.regularization
        factor = cho_factor(gram, lower=True, overwrite_a=True, check_finite=False)
        precision = cho_solve(factor, np.eye(shape[1]), check_finite=False)
        self.weights = precision / (-precision.diagonal()[None, :])
        np.fill_diagonal(self.weights, 0.)
        return self

    def score(self, users):
        return np.asarray(self.history[users] @ self.weights)
