"""Тот же ALS с целевым сигналом из оценки и давности взаимодействия."""

import numpy as np
from .als import ALS


class RecencyALS(ALS):
    def __init__(self, factors=32, regularization=.1, iterations=12,
                 half_life_days=1095., seed=42):
        super().__init__(factors, regularization, iterations, seed)
        self.half_life_days = half_life_days

    def target(self, history, cutoff):
        age = (cutoff - history.timestamp.to_numpy(dtype=float)) / 86_400_000
        assert (age >= 0).all()
        preference = (history.rating.to_numpy(dtype=float) - 3.) / 2.
        return preference * np.exp2(-age / self.half_life_days)

    def fit(self, history, shape, cutoff):
        self.cutoff = cutoff
        return super().fit(history, shape, target=self.target(history, cutoff))
