"""Batched independent towns (town_xp town_size=128, one vectorized call) vs the original CPU path
(town.run_town called once per 128-genome shard). Same genomes; compare per-genome outcome means
over many seeds with a permutation test. Run: python tests/validate_batched_towns.py [--xp fakecupy|cupy]"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genesis import town, town_xp
from genesis.genome import AXES
xp = np
if "--xp" in sys.argv:
    w = sys.argv[sys.argv.index("--xp") + 1]
    if w == "cupy": import cupy as xp
    else: sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from fakecupy import cp as xp
N = int(os.environ.get("NSEEDS", 12)); G = 512; TS = 128
KEYS = ["births", "survival", "honesty", "rep_mean", "bond", "cheat_births", "cheat_survival"]
def perm_p(a, b, n=4000, rng=np.random.default_rng(0)):
    a, b = np.asarray(a), np.asarray(b); d0 = abs(a.mean() - b.mean()); z = np.r_[a, b]; c = 0
    for _ in range(n):
        rng.shuffle(z); c += abs(z[:len(a)].mean() - z[len(a):].mean()) >= d0
    return (c + 1) / (n + 1)
res = {}
for name, rg in (("rung0", dict(T=32, per_genome=2)), ("rung2", dict(T=192, per_genome=6))):
    A = {k: [] for k in KEYS}; Bm = {k: [] for k in KEYS}
    for s in range(N):
        r = np.random.default_rng(1000 + s)
        g = (r.random((G, len(AXES))) * np.array([len(a[1]) for a in AXES])).astype(int)
        env = town.default_env(); env["cheaters"] = 0.1
        outs = [town.run_town(g[i:i + TS], env, 5000 + 31 * s + i, **rg) for i in range(0, G, TS)]
        o_cpu = {k: np.concatenate([o[k] for o in outs]) for k in KEYS}
        o_b = town_xp.run_town(xp.asarray(g), env, 9000 + s, xp=xp, town_size=TS, **rg)
        for k in KEYS: A[k].append(float(np.mean(o_cpu[k]))); Bm[k].append(float(np.mean(o_b[k])))
        print(name, s, "births cpu %.4f batched %.4f" % (A["births"][-1], Bm["births"][-1]), flush=True)
    res[name] = {k: dict(cpu=round(np.mean(A[k]), 5), batched=round(np.mean(Bm[k]), 5), p=round(perm_p(A[k], Bm[k]), 4)) for k in KEYS}
    print(name, json.dumps(res[name]), flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "batched_towns_validation.json"), "w"), indent=1)
