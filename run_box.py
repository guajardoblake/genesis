#!/usr/bin/env python3
"""Run GENESIS on the box. Resumes automatically from out/ckpt.pkl."""
import argparse, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genesis.engine import Genesis
ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=30); ap.add_argument("--out", default="runs/box")
ap.add_argument("--workers", type=int, default=None); ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no-lamarck", action="store_true")
ap.add_argument("--no-speed", action="store_true", help="baseline: no surrogate, no halving, no cache")
a = ap.parse_args()
ck = os.path.join(a.out, "ckpt.pkl")
if os.path.exists(ck): g = Genesis.load(a.out); print("resumed at gen", g.gen)
else: g = Genesis(a.out, seed=a.seed, workers=a.workers, lamarck=not a.no_lamarck, batch=a.batch, speed=not a.no_speed)
g.run(a.minutes)
