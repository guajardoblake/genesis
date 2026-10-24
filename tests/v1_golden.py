"""Deterministic fingerprint of v1 engine output (tiny runs). Generated at commit d7d6f8e, BEFORE v2 was added.
tests/test_v1_unchanged.py re-runs this and requires an exact match, so v2 cannot have changed v1 output."""
import sys, os, json, hashlib, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genesis import engine, town, town_xp, genome as GN
ARMS = {"normal": {}, "ablate": dict(love=False, reputation=False), "cheat": dict(cheaters=0.1)}
def fingerprint():
    out = {}
    for name, ov in ARMS.items():
        with tempfile.TemporaryDirectory() as d:
            g = engine.Genesis(d, seed=5, workers=1, batch=48, shard=16, screen_n=2000, env_overrides=ov)
            rows = []; g.run(1e9, max_gen=3, log=lambda s: rows.append(json.loads(s)))
            for r in rows: r.pop("elapsed_s", None)
            out[name] = hashlib.sha256(json.dumps(rows, sort_keys=True, default=float).encode()).hexdigest()
    rng = np.random.default_rng(1); G = GN.sample(rng, 20)
    for name, ov in ARMS.items():
        env = dict(town.default_env(), **ov)
        for mod in (town, town_xp):
            o = mod.run_town(G, env, 9, T=40, per_genome=3, return_w=True)
            out[f"town_{mod.__name__.split('.')[-1]}_{name}"] = hashlib.sha256(b"".join(np.ascontiguousarray(o[k]).tobytes() for k in sorted(o))).hexdigest()
    return out
if __name__ == "__main__":
    fp = fingerprint(); json.dump(fp, open(os.path.join(os.path.dirname(__file__), "v1_golden.json"), "w"), indent=1); print(fp)
