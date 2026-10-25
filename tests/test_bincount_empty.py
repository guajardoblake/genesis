import numpy as np
from genesis import town_xp, town_xp_v2

class _Strict:
    """numpy stand-in whose bincount raises on empty input, like CuPy."""
    def __getattr__(self, k): return getattr(np, k)
    def bincount(self, x, w=None, minlength=0):
        if np.asarray(x).size == 0: raise ValueError("zero-size array to reduction operation")
        return np.bincount(x, w, minlength=minlength)

def test_empty_matches_numpy():
    for m in (town_xp, town_xp_v2):
        e = np.zeros(0, dtype=np.int64)
        for w in (None, np.zeros(0)):
            got = m._bc(_Strict(), e, w, minlength=5); ref = np.bincount(e, w, minlength=5)
            assert got.shape == ref.shape and got.dtype == ref.dtype and (got == ref).all()

def test_nonempty_identical():
    x = np.array([0, 2, 2, 4]); w = np.array([0.5, 1.0, 2.0, 3.0])
    for m in (town_xp, town_xp_v2):
        assert (m._bc(_Strict(), x, w, minlength=6) == np.bincount(x, w, minlength=6)).all()
