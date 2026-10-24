"""Bayesian linear surrogate on one-hot + pairwise-hashed features. Gives mean and uncertainty
for Thompson/UCB picking; used to pre-screen millions of genomes before any simulation."""
import numpy as np
from .genome import onehot, ONEHOT_DIM, sample

PAIRS = [(0, 1), (0, 11), (7, 8), (2, 3), (6, 11), (1, 9), (5, 2), (10, 7)]
def feats(g):
    from .genome import SIZES
    X = [onehot(g)]
    for a, b in PAIRS:
        k = g[:, a] * SIZES[b] + g[:, b]; m = np.zeros((g.shape[0], SIZES[a] * SIZES[b]), np.float32)
        m[np.arange(g.shape[0]), k] = 1; X.append(m)
    X.append(np.ones((g.shape[0], 1), np.float32))
    return np.concatenate(X, 1)

class Surrogate:
    def __init__(self, alpha=1.0):
        self.alpha = alpha; self.X = []; self.y = []; self.w = None
    def add(self, g, y):
        self.X.append(feats(g)); self.y.append(np.asarray(y, np.float32))
    def fit(self):
        X = np.concatenate(self.X); y = np.concatenate(self.y)
        if len(y) > 200000: X, y = X[-200000:], y[-200000:]
        A = X.T @ X + self.alpha * np.eye(X.shape[1]); self.Ainv = np.linalg.inv(A)
        self.w = (self.Ainv @ (X.T @ y)).astype(np.float32); self.s2 = float(np.var(y - X @ self.w) + 1e-6); self.n = len(y)
    def predict(self, g, std=False):
        X = feats(g); mu = X @ self.w
        if not std: return mu
        return mu, np.sqrt(self.s2 * ((X @ self.Ainv.astype(np.float32)) * X).sum(1))
    def screen(self, rng, n_total, keep, chunk=200000, kappa=1.0):
        """Score n_total random genomes, keep top-`keep` by UCB. Returns genomes, seconds."""
        import time; t0 = time.time(); best = None; bs = None
        for i in range(0, n_total, chunk):
            g = sample(rng, min(chunk, n_total - i)); mu, sd = self.predict(g, True); u = mu + kappa * sd
            if best is None: best, bs = g, u
            else: best = np.concatenate([best, g]); bs = np.concatenate([bs, u])
            o = np.argsort(-bs)[:keep]; best, bs = best[o], bs[o]
        return best, time.time() - t0
