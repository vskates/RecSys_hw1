"""FM второго порядка с BPR loss; градиенты и Adam реализованы явно."""

import numpy as np
from scipy import sparse
from scipy.special import expit


class FactorizationMachine:
    def __init__(self, factors=16, learning_rate=.01, regularization=1e-5,
                 epochs=20, batch_size=2048, seed=42):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def initialize(self, n_users, item_features):
        self.n_users = n_users
        self.item_features = item_features.tocsr()
        count = n_users + item_features.shape[1]
        self.w = np.zeros(count)
        self.v = np.random.default_rng(self.seed).normal(0, .05, (count, self.factors))
        self.mw, self.sw = np.zeros_like(self.w), np.zeros_like(self.w)
        self.mv, self.sv = np.zeros_like(self.v), np.zeros_like(self.v)
        self.steps = 0
        return self

    def pairs(self, users, items):
        users = np.asarray(users)
        onehot = sparse.csr_matrix((np.ones(len(users)), (np.arange(len(users)), users)),
                                  shape=(len(users), self.n_users))
        return sparse.hstack([onehot, self.item_features[items]], format='csr')

    def forward(self, x):
        projected = x @ self.v
        squared = x.multiply(x)
        value = x @ self.w + .5 * (projected**2 - squared @ (self.v**2)).sum(axis=1)
        return value, projected, squared

    def loss_grad(self, positive, negative):
        fp, pp, sp = self.forward(positive)
        fn, pn, sn = self.forward(negative)
        delta = fp - fn
        weight = expit(-delta)
        size = len(delta)
        gw = (-positive.T @ weight + negative.T @ weight) / size
        gv = (-positive.T @ (weight[:, None] * pp)
              + negative.T @ (weight[:, None] * pn)
              + ((sp.T @ weight) - (sn.T @ weight))[:, None] * self.v) / size
        reg = self.regularization
        loss = np.logaddexp(0, -delta).mean() + .5 * reg * (
            np.square(self.w).sum() + np.square(self.v).sum())
        return float(loss), gw + reg*self.w, gv + reg*self.v

    def update(self, gw, gv):
        self.steps += 1
        for parameter, first, second, gradient in [(self.w, self.mw, self.sw, gw),
                                                    (self.v, self.mv, self.sv, gv)]:
            first *= .9
            first += .1 * gradient
            second *= .999
            second += .001 * gradient**2
            parameter -= self.learning_rate * (first / (1-.9**self.steps)) / (
                np.sqrt(second / (1-.999**self.steps)) + 1e-8)

    def fit(self, history, shape, features, evaluator=None, checkpoints=None):
        from utils import matrix
        self.initialize(shape[0], features)
        seen = matrix(history, shape).toarray().astype(bool)
        positives = history[history.rating >= 4]
        u_all = positives.user_idx.to_numpy()
        i_all = positives.item_idx.to_numpy()
        assert not seen.all(axis=1).any()
        rng = np.random.default_rng(self.seed)
        self.trace = []
        self.best_epoch = self.epochs
        best, state = -np.inf, None
        checkpoints = checkpoints or [self.epochs]
        for epoch in range(1, self.epochs + 1):
            total = 0.
            order = rng.permutation(len(positives))
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start+self.batch_size]
                users, items = u_all[batch], i_all[batch]
                negatives = rng.integers(0, shape[1], len(batch))
                bad = seen[users, negatives]
                while bad.any():
                    negatives[bad] = rng.integers(0, shape[1], int(bad.sum()))
                    bad = seen[users, negatives]
                loss, gw, gv = self.loss_grad(self.pairs(users, items),
                                             self.pairs(users, negatives))
                self.update(gw, gv)
                total += loss * len(batch)
            row = {'epoch': epoch, 'BPR_loss': total / len(positives)}
            if evaluator is not None and epoch in checkpoints:
                self._prepare_scores()
                metrics, _ = evaluator.evaluate(self)
                row.update(metrics)
                if metrics['NDCG@10'] > best:
                    best = metrics['NDCG@10']
                    self.best_epoch = epoch
                    state = self.w.copy(), self.v.copy()
                print(f'FM epoch {epoch}: loss={row["BPR_loss"]:.4f}, '
                      f'validation NDCG@10={metrics["NDCG@10"]:.5f}', flush=True)
            self.trace.append(row)
        if state is not None:
            self.w, self.v = state
        self._prepare_scores()
        return self

    def _prepare_scores(self):
        boundary = self.n_users
        self.item_projection = self.item_features @ self.v[boundary:]
        self.item_offset = self.item_features @ self.w[boundary:] + .5 * (
            self.item_projection**2
            - self.item_features.multiply(self.item_features) @ (self.v[boundary:]**2)).sum(axis=1)

    def score(self, users):
        return (self.w[users, None] + self.item_offset[None, :]
                + self.v[users] @ self.item_projection.T)
