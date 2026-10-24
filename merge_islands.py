#!/usr/bin/env python3
"""Merge island checkpoints (box + Kaggle sessions) into one archive: per niche keep the mind that left more offspring."""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genesis.engine import Genesis
dst = Genesis.load(sys.argv[1])
for src in sys.argv[2:]:
    s = Genesis.load(src)
    for a in s.archives:
        for c, r in a.items():
            if c not in dst.archives[0] or r["key"] > dst.archives[0][c]["key"]: dst.archives[0][c] = r
    dst.stats["sim_genomes"] += s.stats["sim_genomes"]; dst.stats["screened"] += s.stats["screened"]
dst.save(); print("merged; env0 archive size", len(dst.archives[0]))
