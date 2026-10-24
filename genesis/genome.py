"""Genome = a recipe for HOW a mind learns, never the finished weights.
Each axis is a discrete choice; the design space is the product of all axis sizes."""
import numpy as np

AXES = [
    ("plasticity_rule",   ["hebb", "oja", "bcm", "anti_hebb", "eligibility_hebb", "predictive_delta"]),
    ("learning_rate",     [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]),
    ("memory_type",       ["none", "leaky_trace", "fast_weights", "episodic_slots"]),
    ("memory_decay",      [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]),
    ("prediction_horizon",[1, 2, 4, 8]),
    ("hidden_units",      [4, 8, 12, 16]),
    ("surprise_gain",     [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]),     # internal free-energy weight into hidden balance
    ("love_wiring",       ["none", "to_balance", "to_reproduction", "to_both", "to_both_strong"]),
    ("partner_pred_weight",[0.0, 0.25, 0.5, 1.0, 2.0]),           # how much partner's inner sense is predicted/learned
    ("exploration",       [0.0, 0.05, 0.1, 0.2, 0.4]),
    ("trust_sensitivity", [0.0, 0.25, 0.5, 1.0, 2.0]),           # how much others' reputation shapes approach
    ("neuromod_source",   ["constant", "surprise_drop", "balance_gain", "bond_gain"]),
]
NAMES = [a[0] for a in AXES]
SIZES = np.array([len(a[1]) for a in AXES], dtype=np.int64)
N_AXES = len(AXES)
SPACE_SIZE = int(np.prod(SIZES))
ONEHOT_DIM = int(SIZES.sum())
_OFF = np.concatenate([[0], np.cumsum(SIZES)[:-1]])

def value(g, axis):
    i = NAMES.index(axis); return AXES[i][1][g[i]]

def sample(rng, n):
    return (rng.random((n, N_AXES)) * SIZES).astype(np.int64)

def mutate(rng, g, rate=0.15):
    g = g.copy()
    m = rng.random(g.shape) < rate
    # guarantee at least one change per genome
    rows = np.arange(g.shape[0]); m[rows, rng.integers(0, N_AXES, g.shape[0])] = True
    step = rng.choice([-1, 1], size=g.shape)
    jump = rng.random(g.shape) < 0.3
    new = np.where(jump, (rng.random(g.shape) * SIZES).astype(np.int64), np.clip(g + step, 0, SIZES - 1))
    g[m] = new[m]
    return g

def crossover(rng, a, b):
    mask = rng.random(a.shape) < 0.5
    return np.where(mask, a, b)

def onehot(g):
    X = np.zeros((g.shape[0], ONEHOT_DIM), dtype=np.float32)
    X[np.arange(g.shape[0])[:, None], g + _OFF] = 1.0
    return X

def to_index(g):
    idx = np.zeros(g.shape[0], dtype=np.int64)
    for i in range(N_AXES): idx = idx * SIZES[i] + g[:, i]
    return idx

def describe(g):
    return {n: AXES[i][1][int(g[i])] for i, n in enumerate(NAMES)}
