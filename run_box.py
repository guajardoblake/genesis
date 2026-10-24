#!/usr/bin/env python3
"""Run GENESIS on the box. Resumes automatically from out/ckpt.pkl."""
import argparse, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genesis.engine import Genesis
ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=30); ap.add_argument("--out", default="runs/box")
ap.add_argument("--workers", type=int, default=None); ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--max-gen", type=int, default=None, help="stop at this generation (resume-safe)")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no-lamarck", action="store_true")
ap.add_argument("--no-speed", action="store_true", help="baseline: no surrogate, no halving, no cache")
ap.add_argument("--no-love", action="store_true", help="ablation: bond never feeds balance/reproduction drive")
ap.add_argument("--no-rep", action="store_true", help="ablation: reputation fixed at 0.5 (no trust signal)")
ap.add_argument("--cheaters", type=float, default=0.0, help="extra fraction of fake-reputation cheater minds")
ap.add_argument("--rules", default="v1", choices=["v1", "v2"], help="v2 = fixed mechanics (genesis/town_v2.py)")
ap.add_argument("--cheat-mode", default="standard", choices=["standard", "stranger", "evolve", "planted_honest"], help="v2 cheater arm")
ap.add_argument("--long", action="store_true", help="longer lives (final rung 512 steps) and bigger town (size 36)")
a = ap.parse_args()
ov = {}
if a.no_love: ov["love"] = False
if a.no_rep: ov["reputation"] = False
if a.cheaters: ov["cheaters"] = a.cheaters
if a.rules == "v2": ov.update(rules="v2", cheat_mode=a.cheat_mode)
if a.long: ov.update(t_scale=8 / 3, size=36.0)
ck = os.path.join(a.out, "ckpt.pkl")
if os.path.exists(ck): g = Genesis.load(a.out); print("resumed at gen", g.gen)
else: g = Genesis(a.out, seed=a.seed, workers=a.workers, lamarck=not a.no_lamarck, batch=a.batch, speed=not a.no_speed, env_overrides=ov)
g.run(a.minutes, max_gen=a.max_gen)
