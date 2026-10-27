"""Item-CF: adjusted cosine, overlap shrinkage и усреднение отклонений."""

import numpy as np
from scipy import sparse


class ItemKNN:
    def __init__(self, neighbors=100, shrinkage=10., min_common=3):
        self.neighbors = neighbors
        self.shrinkage = shrinkage
        self.min_common = min_common

    def fit(self, history, shape):
        from utils import matrix
        ratings = matrix(history, shape, history.rating.to_numpy())
        self.observed = matrix(history, shape)
        count = self.observed.getnnz(axis=1)
        self.means = np.divide(np.asarray(ratings.sum(axis=1)).ravel(), count,
                               out=np.zeros(shape[0]), where=count > 0)
        centered_values = history.rating.to_numpy() - self.means[history.user_idx.to_numpy()]
        self.centered = matrix(history, shape, centered_values)
        gram = (self.centered.T @ self.centered).toarray()
        norm = np.sqrt(np.maximum(gram.diagonal(), 0))
        denom = norm[:, None] * norm[None, :]
        np.divide(gram, denom, out=gram, where=denom > 0)
        gram[denom == 0] = 0
        del denom
        overlap = (self.observed.T @ self.observed).toarray()
        gram *= overlap / (overlap + self.shrinkage)
        gram[overlap < self.min_common] = 0
        del overlap
        np.fill_diagonal(gram, 0)
        gram[gram < 0] = 0
        # Столбец = candidate, строки = его соседи из истории.
        rows, cols, vals = [], [], []
        for j in range(shape[1]):
            top = np.argsort(-gram[:, j], kind='stable')[:self.neighbors]
            top = top[gram[top, j] > 0]
            rows.extend(top); cols.extend([j] * len(top)); vals.extend(gram[top, j])
        self.similarity = sparse.csc_matrix((vals, (rows, cols)), shape=gram.shape)
        return self

    def score(self, users):
        numerator = (self.centered[users] @ self.similarity).toarray()
        denominator = (self.observed[users] @ self.similarity).toarray()
        deviation = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                               where=denominator > 0)
        return self.means[users, None] + deviation
