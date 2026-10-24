#!/usr/bin/env python3
"""Compare multi-seed GENESIS runs. Usage: python3 analyze_seeds.py runs/seed0 runs/seed1 ... > MULTISEED.json"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genesis import genome as GN
from genesis.engine import Genesis

def summarize(out):
    P = [json.loads(l) for l in open(os.path.join(out, "progress.jsonl"))]
    seen = {}; [seen.__setitem__(p["gen"], p) for p in P]; P = [seen[k] for k in sorted(seen)]   # dedupe on resume
    L = [json.loads(l) for l in open(os.path.join(out, "lineage.jsonl"))]
    g = Genesis.load(out); last = P[-1]
    base = [p["law"][0] for p in P if p.get("law")]
    allw = [w for p in P[-20:] if p.get("law") for w in p["law"] if w and w["promise_keeping"] is not None]
    def m(rows, k): v = [r[k] for r in rows if r.get(k) is not None]; return float(np.mean(v)) if v else None
    half = L[len(L) // 2:]; feats = {}
    for name, vals in GN.AXES:
        sh = [r[name] for r in half if name in r]
        if not sh: continue
        mm = {str(v): float(np.mean([s[str(v)] for s in sh])) for v in vals}; best = max(mm, key=mm.get)
        lead = sum(1 for r in L if name in r and max(r[name], key=r[name].get) == best)
        feats[name] = dict(lead=best, share=round(mm[best], 4), lift=round(mm[best] * len(vals), 2), gens_leading=lead, of=len(L))
    tops = []
    for a in g.archives:
        best = sorted(a.values(), key=lambda r: -r["key"])[:3]
        tops.append(dict(births=round(float(np.mean([r["o"]["births"] for r in best])), 3),
                         law_breaks=round(float(max(r["o"]["law_breaks"] for r in best)), 4)))
    return dict(run=out, gens=g.gen, worlds=len(g.envs), niches=sum(len(a) for a in g.archives), stats=g.stats,
                births_gen1=P[0]["mean_births_final_rung"][0], births_final_base=last["mean_births_final_rung"][0],
                births_final_all_worlds=last["mean_births_final_rung"],
                law_base_first20=dict(promise_keeping=m(base[:20], "promise_keeping"), honesty=m(base[:20], "honesty"), rep_mean=m(base[:20], "rep_mean"), rep_gap=m(base[:20], "rep_gap_mean"), corr=m(base[:20], "rep_honesty_corr")),
                law_base_last20=dict(promise_keeping=m(base[-20:], "promise_keeping"), honesty=m(base[-20:], "honesty"), rep_mean=m(base[-20:], "rep_mean"), rep_gap=m(base[-20:], "rep_gap_mean"), corr=m(base[-20:], "rep_honesty_corr")),
                law_allworlds_last20=dict(promise_keeping=m(allw, "promise_keeping"), honesty=m(allw, "honesty"), rep_gap=m(allw, "rep_gap_mean"), corr=m(allw, "rep_honesty_corr")),
                features=feats, top_minds=tops)

if __name__ == "__main__":
    print(json.dumps([summarize(o) for o in sys.argv[1:]], indent=1, default=float))
