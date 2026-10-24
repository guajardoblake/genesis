"""Unit tests for law-keeping metrics (promise-keeping rate, reputation gap). Run: python3 -m unittest -v"""
import os, sys, math, unittest, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genesis import genome as GN
from genesis.town import run_town, default_env, promise_keeping_rate, honesty, reputation_gap
from genesis.engine import law_metrics


class PureFunctions(unittest.TestCase):
    def test_promise_keeping(self):
        self.assertEqual(promise_keeping_rate([0, 0], [5, 5]), 1.0)
        self.assertAlmostEqual(promise_keeping_rate([1, 2], [4, 6]), 0.7)
        self.assertEqual(promise_keeping_rate([3], [3]), 0.0)
        self.assertTrue(math.isnan(promise_keeping_rate([0], [0])))

    def test_honesty(self):
        np.testing.assert_allclose(honesty([0, 2, 5], [10, 4, 5]), [1.0, 0.5, 0.0])
        np.testing.assert_allclose(honesty([0], [0]), [1.0])   # no encounters -> nothing broken

    def test_reputation_gap_sign(self):
        g = reputation_gap([0.9, 0.2, 0.5], [0.5, 0.6, 0.5])
        np.testing.assert_allclose(g, [0.4, -0.4, 0.0])        # >0 means reputation overstates conduct


class TownInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1)
        cls.g = GN.sample(rng, 64)
        cls.o = run_town(cls.g, default_env(), seed=7, T=96, per_genome=4)

    def test_fields_present_and_bounded(self):
        o = self.o
        for k in ("promise_opps", "promise_breaks", "encounters", "law_break_events", "rep_mean", "honesty"):
            self.assertEqual(o[k].shape, (64,))
            self.assertTrue(np.all(np.isfinite(o[k])))
        self.assertTrue(np.all((o["honesty"] >= 0) & (o["honesty"] <= 1)))
        self.assertTrue(np.all((o["rep_mean"] >= 0) & (o["rep_mean"] <= 1)))

    def test_breaks_subset_of_opportunities(self):
        o = self.o
        self.assertTrue(np.all(o["promise_breaks"] <= o["promise_opps"] + 1e-9))
        self.assertTrue(np.all(o["law_break_events"] <= o["encounters"] + 1e-9))

    def test_world_summary(self):
        m = law_metrics(self.o)
        self.assertEqual(m["n"], 64)
        if m["promise_keeping"] is not None:
            self.assertTrue(0 <= m["promise_keeping"] <= 1)
        self.assertAlmostEqual(m["rep_gap_mean"], m["rep_mean"] - m["honesty"], places=4)

    def test_instrumentation_does_not_change_dynamics(self):
        # same seed -> identical births (metrics add no RNG draws)
        o2 = run_town(self.g, default_env(), seed=7, T=96, per_genome=4)
        np.testing.assert_array_equal(o2["births"], self.o["births"])


if __name__ == "__main__":
    unittest.main()


class Experiments(unittest.TestCase):
    def setUp(self):
        self.g = GN.sample(np.random.default_rng(2), 32)

    def test_cheaters_reported_and_stripped(self):
        o = run_town(self.g, dict(default_env(), cheaters=0.25), seed=3, T=64, per_genome=2, return_w=True)
        self.assertEqual(o["births"].shape, (32,)); self.assertEqual(o["Win"].shape[0], 32)
        self.assertAlmostEqual(o["cheat_frac"][0], 8 / 40)
        self.assertTrue(0 <= o["cheat_bond_share"][0] <= 1)

    def test_rep_ablation_flattens_reputation(self):
        o = run_town(self.g, dict(default_env(), reputation=False), seed=3, T=64, per_genome=2)
        np.testing.assert_allclose(o["rep_mean"][o["rep_mean"] > 0], 0.5)

    def test_long_rungs(self):
        from genesis.engine import rung_cfg
        self.assertEqual(rung_cfg(dict(t_scale=8 / 3), 2)["T"], 512)
        self.assertEqual(rung_cfg({}, 2)["T"], 192)
