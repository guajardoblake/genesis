"""JAX vs numpy parity on fixed seeds (CPU). Skipped if jax is not installed."""
import os, sys, unittest, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import jax.numpy as jnp
    from genesis import core_jax as CJ
except ImportError:
    CJ = None
from genesis import core_np as CN

def inputs(seed, B=256, H=16, D=14, A=8):
    r = np.random.default_rng(seed); f = lambda *s, sc=1.0: (r.normal(0, sc, s)).astype(np.float32)
    return dict(Win=f(B, H, D, sc=0.4), Wout=f(B, A, H, sc=0.1), obs=f(B, D), mem_t=r.integers(0, 4, B), trace=f(B, H),
                fw=f(B, H, H, sc=0.1), slots=f(B, 4, H), hmask=(np.arange(H)[None] < r.choice([4, 8, 12, 16], B)[:, None]).astype(np.float32),
                lrp=r.choice([0, 0, 1, 1, 2.0], B), bond=r.random(B), ex=r.choice([0, .05, .1, .2, .4], B), u=r.random(B))

@unittest.skipIf(CJ is None, "jax not installed")
class Parity(unittest.TestCase):
    def test_mind_forward(self):
        for seed in range(5):
            x = inputs(seed); hn, pn, an = CN.mind_forward(**x); hj, pj, aj = CJ.mind_forward(**x)
            np.testing.assert_allclose(np.asarray(hj), hn, atol=1e-5); np.testing.assert_allclose(np.asarray(pj), pn, atol=1e-6)
            np.testing.assert_array_equal(np.asarray(aj), an)

    def test_predict_sense(self):
        r = np.random.default_rng(9); B = 256
        args = (r.normal(0, .1, (B, 6, 16)).astype(np.float32), r.normal(0, 1, (B, 16)).astype(np.float32), r.random((B, 6)).astype(np.float32),
                r.random((B, 6)).astype(np.float32), r.choice([1, 2, 4, 8], B).astype(float), r.choice([0, .25, .5, 1, 2.0], B))
        for n, j in zip(CN.predict_sense(*args), CJ.predict_sense(*args)):
            np.testing.assert_allclose(np.asarray(j), n, rtol=1e-5, atol=1e-6)

    def test_law_step(self):
        r = np.random.default_rng(4); B = 512
        partner = np.where(r.random(B) < .5, r.integers(0, B, B), -1)
        near = np.where(r.random(B) < .5, np.maximum(partner, 0), r.integers(0, B, B))
        args = (r.integers(0, 8, B), r.random(B) < .9, r.random(B) < .6, near, partner, r.random(B) < .7, r.random(B))
        for n, j in zip(CN.law_step(*args), CJ.law_step(*args)):
            np.testing.assert_allclose(np.asarray(j), n, atol=1e-12)
