"""town_xp (numpy backend) must be bit-identical to town.run_town."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genesis import genome as GN, town, town_xp
def test_bit_identical():
    rng = np.random.default_rng(3); g = GN.sample(rng, 24)
    for env in (town.default_env(), dict(town.default_env(), cheaters=0.1), dict(town.default_env(), reputation=False, love=False)):
        a = town.run_town(g, env, 11, T=48, per_genome=3, return_w=True); b = town_xp.run_town(g, env, 11, T=48, per_genome=3, return_w=True)
        assert a.keys() == b.keys()
        for k in a: assert np.array_equal(a[k], b[k], equal_nan=True), k
if __name__ == "__main__": test_bit_identical(); print("bit-identical OK")
