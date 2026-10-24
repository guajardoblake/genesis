#!/usr/bin/env python3
"""'Most likely answer remaining' report: which design choices keep surviving across generations."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genesis import genome as GN
from genesis.engine import Genesis
out = sys.argv[1] if len(sys.argv) > 1 else "runs/box"
rows = [json.loads(l) for l in open(os.path.join(out, "lineage.jsonl"))]
g = Genesis.load(out); L = ["# GENESIS lineage report", f"run: {out}, generations: {g.gen}, worlds: {len(g.envs)}", ""]
L.append(f"stats: {json.dumps(g.stats)}"); L.append("")
L.append("## Worlds (POET co-evolved)"); [L.append(f"- env{i}: " + json.dumps({k: round(v, 4) for k, v in e.items()}) + f", archive niches: {len(g.archives[i])}") for i, e in enumerate(g.envs)]
half = rows[len(rows) // 2:]
L += ["", "## Design features carried by reproducers (offspring-weighted share, second half of run vs uniform)", ""]
L.append("| axis | leading value | share | uniform | lift | gens leading (of %d) |" % len(rows))
L.append("|---|---|---|---|---|---|")
for name, vals in GN.AXES:
    sh = [r[name] for r in half if name in r]
    if not sh: continue
    m = {v: float(np.mean([s[str(v)] for s in sh])) for v in vals}
    best = max(m, key=m.get); u = 1 / len(vals)
    lead = sum(1 for r in rows if name in r and max(r[name], key=r[name].get) == str(best))
    L.append(f"| {name} | {best} | {m[best]:.3f} | {u:.3f} | {m[best]/u:.2f}x | {lead} |")
L += ["", "## Top minds per world (by offspring)"]
for i, a in enumerate(g.archives):
    for r in sorted(a.values(), key=lambda r: -r["key"])[:3]:
        o = r["o"]; L.append(f"- env{i} births/mind {o['births']:.3f}, survival {o['survival']:.2f}, balanced {o['balanced']:.2f}, bond {o['bond']:.2f}, law-breaks/step {o['law_breaks']:.3f}, tools {o['tools']:.2f}: {GN.describe(r['g'])}")
open(os.path.join(out, "REPORT.md"), "w").write("\n".join(L)); print("\n".join(L))
