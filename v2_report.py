#!/usr/bin/env python3
"""v2 rerun statistics (runs_v2/{a,b,c,d}/runs/*): paired normal vs ablate (control), cheater arms vs planted_honest.
Usage: python3 v2_report.py -> runs_v2/v2_summary.json"""
import os, json, glob, numpy as np
from multiseed_report import ci, prog, window, base_law, births0
def find(arm): return {int(d.rsplit("seed",1)[1]): d for d in glob.glob(f"runs_v2/*/runs/v2_{arm}_seed*") if os.path.isdir(d) and os.path.exists(os.path.join(d,"DONE"))}
R = {}
N, A = find("normal"), find("ablate")
def f(P, g0=91, g1=100):
    return dict(births=window(P,g0,g1,births0), births_all_worlds=window(P,g0,g1,lambda p: float(np.mean(p["mean_births_final_rung"]))),
                promise_keeping=window(P,g0,g1,base_law("promise_keeping")), honesty=window(P,g0,g1,base_law("honesty")),
                rep_honesty_corr=window(P,g0,g1,base_law("rep_honesty_corr")), rep_gap_cal=window(P,g0,g1,base_law("rep_gap_cal")),
                niches=sum(P[max(P)]["archive"]), worlds=P[max(P)]["envs"], gens=max(P))
pair = {s: dict(main=f(prog(N[s])), control=f(prog(A[s]))) for s in sorted(set(N)&set(A))}
R["paired_per_seed"] = pair
R["paired_summary"] = {k: dict(main=ci([p["main"][k] for p in pair.values()]), control=ci([p["control"][k] for p in pair.values()]),
    diff_main_minus_control=ci([None if p["main"][k] is None or p["control"][k] is None else p["main"][k]-p["control"][k] for p in pair.values()]),
    seeds_main_higher=sum(1 for p in pair.values() if p["main"][k] is not None and p["control"][k] is not None and p["main"][k] > p["control"][k]))
    for k in next(iter(pair.values()))["main"]} if pair else {}
arms = {}
for arm in ("cheat_stranger","cheat_evolve","planted_honest"):
    rows = {}
    for s, d in sorted(find(arm).items()):
        P = prog(d); L = lambda k: (lambda p: (p.get("law") or [None])[0] and p["law"][0].get(k))
        keys = sorted(set(k for p in P.values() if p.get("law") for k in p["law"][0] if k.startswith("cheat") or k.startswith("c_") or k.startswith("honest")))
        r = {k+"_last20": window(P,81,100,L(k)) for k in keys}
        r.update({k+"_first20": window(P,1,20,L(k)) for k in keys if "share" in k})
        r.update(births_last20=window(P,81,100,births0), promise_keeping_last20=window(P,81,100,L("promise_keeping")), honesty_last20=window(P,81,100,L("honesty")))
        rows[s] = r
    arms[arm] = dict(per_seed=rows, summary={k: ci([r.get(k) for r in rows.values()]) for k in (next(iter(rows.values())) if rows else {})})
R["cheater_arms"] = arms
json.dump(R, open("runs_v2/v2_summary.json","w"), indent=1)
