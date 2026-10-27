"""ALS для явных наблюдений; пропуски не входят в функцию потерь."""

import numpy as np
from scipy import sparse


class ALS:
    def __init__(self, factors=32, regularization=5., iterations=12, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed

    def fit(self, history, shape, target=None):
        values = history.rating.to_numpy(dtype=float) if target is None else np.asarray(target)
        observed = sparse.csr_matrix((values, (history.user_idx, history.item_idx)), shape=shape)
        # Нулевые targets сохранены как наблюдения.
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, .1, (shape[0], self.factors))
        self.item_factors = rng.normal(0, .1, (shape[1], self.factors))
        item_rows = observed.T.tocsr()
        ridge = self.regularization * np.eye(self.factors)
        self.losses = []
        for _ in range(self.iterations):
            self._step(observed, self.item_factors, self.user_factors, ridge)
            self._step(item_rows, self.user_factors, self.item_factors, ridge)
            u, i = history.user_idx.to_numpy(), history.item_idx.to_numpy()
            residual = values - np.einsum('ij,ij->i', self.user_factors[u], self.item_factors[i])
            loss = residual @ residual + self.regularization * (
                np.square(self.user_factors).sum() + np.square(self.item_factors).sum())
            self.losses.append(float(loss))
        assert np.isfinite(self.losses).all()
        return self

    @staticmethod
    def _step(rows, fixed, result, ridge):
        for row in range(rows.shape[0]):
            start, end = rows.indptr[row:row + 2]
            factors = fixed[rows.indices[start:end]]
            result[row] = np.linalg.solve(factors.T @ factors + ridge,
                                           factors.T @ rows.data[start:end])

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
