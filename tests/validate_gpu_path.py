"""Statistical equivalence: town_xp CuPy code path (via fakecupy on CPU; different RNG stream) vs numpy town.run_town.
For each config, N independent seeds; compare per-seed town means with two-sided permutation test."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakecupy import cp
from genesis import genome as GN, town, town_xp
def perm_p(x, y, n=4000, rng=np.random.default_rng(0)):
    x, y = np.asarray(x), np.asarray(y); obs = abs(x.mean() - y.mean()); z = np.concatenate([x, y]); c = 0
    for _ in range(n):
        rng.shuffle(z); c += abs(z[:len(x)].mean() - z[len(x):].mean()) >= obs
    return (c + 1) / (n + 1)
N = int(os.environ.get("NSEEDS", 24)); KEYS = ["births", "survival", "balanced", "honesty", "law_breaks", "coop", "rep_mean", "bond", "tools"]
res = {}
for name, env in [("default", town.default_env()), ("cheaters0.1", dict(town.default_env(), cheaters=0.1))]:
    A = {k: [] for k in KEYS}; B = {k: [] for k in KEYS}
    for s in range(N):
        g = GN.sample(np.random.default_rng(1000 + s), 64)
        a = town.run_town(g, env, s, T=96, per_genome=4); b = town_xp.run_town(g, env, s, T=96, per_genome=4, xp=cp)
        for k in KEYS: A[k].append(float(np.nanmean(a[k]))); B[k].append(float(np.nanmean(b[k])))
    for k in KEYS:
        p = perm_p(A[k], B[k])
        d = (np.mean(B[k]) - np.mean(A[k])) / (np.sqrt((np.var(A[k]) + np.var(B[k])) / 2) + 1e-12)
        res[f"{name}/{k}"] = dict(numpy=round(np.mean(A[k]), 4), gpu_path=round(np.mean(B[k]), 4), p=round(float(p), 3), cohen_d=round(float(d), 3))
bad = [k for k, v in res.items() if v["p"] < 0.01]
for k, v in res.items(): print(f"{k:28s} np={v['numpy']:.4f} gpu={v['gpu_path']:.4f} p={v['p']:.3f} d={v['cohen_d']:+.2f}")
print("N seeds", N, "| tests", len(res), "| p<0.01:", bad or "none")
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_path_validation.json"), "w"), indent=1)
