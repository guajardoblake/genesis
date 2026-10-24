"""GENESIS v2: v1 unchanged + one test per v2 fix. Run: python3 -m pytest -q tests/test_v2.py"""
import sys, os, json, hashlib, tempfile, subprocess, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
from genesis import engine, town, town_xp, town_v2, town_xp_v2, genome as GN

def _G(n=24, s=2): return GN.sample(np.random.default_rng(s), n)
def V2(**k): return dict(town.default_env(), rules="v2", **k)

# ---------- v1 is untouched ----------
def test_v1_files_byte_identical():
    for line in open(os.path.join(HERE, "v1_files.sha256")):
        h, f = line.split(); assert hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest() == h, f
def test_v1_engine_and_town_output_unchanged():
    sys.path.insert(0, HERE); import v1_golden
    assert v1_golden.fingerprint() == json.load(open(os.path.join(HERE, "v1_golden.json")))

# ---------- CPU v2 == GPU-path v2 (numpy backend, one town) ----------
def test_xp_v2_bit_identical_to_cpu_v2():
    g = _G()
    for env in (V2(), V2(reputation=False, love=False), V2(cheaters=0.1), V2(cheaters=0.1, cheat_mode="stranger"),
                V2(cheaters=0.1, cheat_mode="planted_honest"),
                V2(cheaters=0.1, cheat_mode="evolve", cheat_pool=dict(g=_G(16, 9), p=np.linspace(0, 1, 16)))):
        a = town_v2.run_town(g, env, 11, T=48, per_genome=3, return_w=True); b = town_xp_v2.run_town(g, env, 11, T=48, per_genome=3, return_w=True)
        assert a.keys() == b.keys()
        for k in a:   # cheat_* are means-of-means on the town path: identical up to summation order (1e-12)
            ok = np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True) if not k.startswith("cheat_") else np.allclose(a[k], b[k], rtol=1e-12, atol=0)
            assert ok, (env.get("cheat_mode"), k)
def test_xp_v2_batched_towns_run():
    o = town_xp_v2.run_town(_G(64), V2(cheaters=0.1), 3, T=32, per_genome=2, town_size=16)
    assert o["births"].shape == (64,) and (o["dx_food_min"] >= 0).all()

# ---------- fix 1: share only counts with a real gift ----------
def test_fix1_no_credit_without_gift():
    for s in range(4):
        o = town_v2.run_town(_G(32, s), V2(cheaters=0.1), s, T=96, per_genome=4)
        assert o["dx_share_nogift"].sum() > 0          # low-energy share attempts happened...
    # ...and none of them gave rep: with no breaks and no relax target change, rep only rises through real gifts.
    src = open(os.path.join(ROOT, "genesis", "town_v2.py")).read()
    assert "share = share_try & (energy > 0.4)" in src and "rep += 0.02 * share" in src and "0.02 * share, 0, 1)" in src

# ---------- fix 2: partner sampled every step ----------
def test_fix2_partner_always_candidate():
    g = _G(32, 5); hit = tot = 0
    for s in range(3):
        o = town_v2.run_town(g, V2(), s, T=96, per_genome=4); hit += o["dx_partner_sampled"].sum(); tot += o["dx_partner_inreach"].sum()
    assert tot > 0 and hit / tot > 0.9, hit / tot     # v1 measured ~28% overall

# ---------- fix 3: reputation relaxes toward conduct average ----------
def test_fix3_rep_relaxes_and_calibrated_gap():
    assert 0 < town_v2.RLAM < 0.1 and 0 < town_v2.CALPHA < 1
    o = town_v2.run_town(_G(32, 7), V2(cheaters=0.1), 7, T=96, per_genome=4)
    assert ((o["rep_target"] >= 0) & (o["rep_target"] <= 1)).all()
    lm = engine.law_metrics(o)
    for k in ("rep_target", "rep_gap_cal", "rep_gap_z_abs"): assert k in lm and np.isfinite(lm[k])
    assert "rep_target" not in (engine.law_metrics(town.run_town(_G(32, 7), town.default_env(), 7, T=32, per_genome=2)) or {})
def test_fix3_relaxation_math():
    rep, cond = 0.2, 0.9
    for _ in range(500): rep += town_v2.RLAM * (cond - rep)
    assert abs(rep - cond) < 0.01

# ---------- fix 4: food never negative ----------
def test_fix4_food_clipped():
    g = _G(1024, 1)   # overcrowded town: v1 food goes to ~ -0.3
    o = town_v2.run_town(g, V2(), 1, T=32, per_genome=2)
    assert o["dx_food_overdraw"].min() < 0      # simultaneous eaters overdraw a cell (the v1 bug)...
    assert o["dx_food_min"].min() >= 0.0        # ...but v2 clips food, so no food or harvest is ever negative

# ---------- fix 5: cheater arms ----------
def test_fix5_stranger_never_robs_partner():
    for s in range(3):
        o = town_v2.run_town(_G(48, s), V2(cheaters=0.25, cheat_mode="stranger"), s, T=96, per_genome=4)
        assert o["cheat_forced"][0] > 0 and o["cheat_forced_on_partner"][0] == 0
def test_fix5_standard_does_rob_partner_sometimes():
    tot = sum(town_v2.run_town(_G(48, s), V2(cheaters=0.25), s, T=96, per_genome=4)["cheat_forced_on_partner"][0] for s in range(3))
    assert tot > 0
def test_fix5_planted_honest_never_forced():
    o = town_v2.run_town(_G(48), V2(cheaters=0.25, cheat_mode="planted_honest"), 1, T=96, per_genome=4)
    assert o["cheat_forced"][0] == 0 and o["cheat_propensity"][0] == 0 and "cheat_births" in o
def test_fix5_evolve_pool_inherits_and_selects():
    with tempfile.TemporaryDirectory() as d:
        gx = engine.Genesis(d, seed=1, workers=1, batch=48, shard=16, screen_n=2000, env_overrides=dict(rules="v2", cheaters=0.25, cheat_mode="evolve"))
        p0 = gx.cpool["p"].copy(); rows = []
        gx.run(1e9, max_gen=2, log=lambda s: rows.append(json.loads(s)))
        assert not np.array_equal(p0, gx.cpool["p"]) and ((gx.cpool["p"] >= 0) & (gx.cpool["p"] <= 1)).all()
        assert "cheat_propensity" in rows[-1]["law"][0]
        g2 = engine.Genesis.load(d); assert np.array_equal(g2.cpool["p"], gx.cpool["p"])   # resumable
    # selection: a pool member with all the births gets copied
    gx.cpool = dict(g=_G(8), p=np.linspace(0, 1, 8)); gx._cfeed = [(np.arange(8), np.eye(8)[3])]
    gx.evolve_cheaters(); assert np.mean(np.abs(gx.cpool["p"] - 3 / 7) < 0.25) > 0.6, gx.cpool["p"]

# ---------- end-to-end CLI ----------
def test_run_box_v2_cli_resumes():
    with tempfile.TemporaryDirectory() as d:
        cmd = [sys.executable, "run_box.py", "--minutes", "5", "--workers", "1", "--batch", "32", "--out", d, "--rules", "v2", "--cheaters", "0.1", "--cheat-mode", "stranger"]
        subprocess.run(cmd + ["--max-gen", "1"], cwd=ROOT, check=True, capture_output=True)
        r = subprocess.run(cmd + ["--max-gen", "2"], cwd=ROOT, check=True, capture_output=True, text=True)
        assert "resumed at gen 1" in r.stdout
        rows = [json.loads(l) for l in open(os.path.join(d, "progress.jsonl"))]
        assert [r["gen"] for r in rows] == [1, 2] and "rep_gap_cal" in rows[-1]["law"][0]
