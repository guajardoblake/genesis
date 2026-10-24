#!/usr/bin/env python3
"""Measure real throughput: baseline (1 core, every genome gets a full life) vs speed layer."""
import os, sys, time, json
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from genesis.genome import sample
from genesis.town import run_town, default_env
from genesis.engine import RUNGS, _work
from genesis.surrogate import Surrogate
rng = np.random.default_rng(1); env = default_env(); R = {}
def timed(f):
    t = time.time(); f(); return time.time() - t
# baseline: one core, full-fidelity life for every genome
g = sample(rng, 256); dt = timed(lambda: run_town(g, env, 0, **RUNGS[2])); R["baseline_full_life_1core_genomes_per_h"] = 256 / dt * 3600
for r in range(3):
    g = sample(rng, 512); dt = timed(lambda: run_town(g, env, 0, **RUNGS[r])); R[f"rung{r}_1core_genomes_per_h"] = 512 / dt * 3600
W = os.cpu_count() - 1
with Pool(W) as p:
    for r in range(3):
        g = sample(rng, 128 * W * 4); jobs = [(g[i:i+128], env, i, r, None) for i in range(0, len(g), 128)]
        dt = timed(lambda: list(p.imap(_work, jobs))); R[f"rung{r}_{W}workers_genomes_per_h"] = len(g) / dt * 3600
# successive halving cost per candidate: 1 rung0 + 0.34 rung1 + 0.1156 rung2
c = 1 / R[f"rung0_{W}workers_genomes_per_h"] + 0.34 / R[f"rung1_{W}workers_genomes_per_h"] + 0.1156 / R[f"rung2_{W}workers_genomes_per_h"]
R["halving_candidates_triaged_per_h"] = 1 / c
s = Surrogate(); gg = sample(rng, 4000); s.add(gg, rng.random(4000)); s.fit()
_, dt = s.screen(rng, 2_000_000, 1000); R["surrogate_screens_per_h"] = 2e6 / dt * 3600
R["hours_for_10M_surrogate_screens"] = 1e7 / R["surrogate_screens_per_h"]
R["hours_for_10M_triaged_sim_candidates"] = 1e7 / R["halving_candidates_triaged_per_h"]
R["speedup_triage_vs_baseline"] = R["halving_candidates_triaged_per_h"] / R["baseline_full_life_1core_genomes_per_h"]
R["speedup_full_life_parallel_vs_baseline"] = R[f"rung2_{W}workers_genomes_per_h"] / R["baseline_full_life_1core_genomes_per_h"]
R["speedup_screening_vs_baseline"] = R["surrogate_screens_per_h"] / R["baseline_full_life_1core_genomes_per_h"]
print(json.dumps({k: round(v, 3) for k, v in R.items()}, indent=1))
json.dump(R, open("bench_result.json", "w"), indent=1)
