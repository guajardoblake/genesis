#!/usr/bin/env python3
"""Side-by-side v1 vs v2: all-worlds and starting-world honesty, matched pairs, cheater offspring. -> runs_v2/v1_v2_compare.json"""
import os, json, glob, numpy as np
from multiseed_report import ci
def prog(d):
    P = {}
    for l in open(os.path.join(d, "progress.jsonl")): p = json.loads(l); P[p["gen"]] = p
    return P
def win(P, g0, g1, k, allw=True):
    v = []
    for g in range(g0, g1 + 1):
        if g not in P or not P[g].get("law"): continue
        rows = P[g]["law"] if allw else P[g]["law"][:1]
        x = [r.get(k) for r in rows if r and r.get(k) is not None]
        if x: v.append(np.mean(x))
    return float(np.mean(v)) if v else None
def arm(pat):
    return {int(d.rstrip('/').rsplit("seed",1)[1]): d for d in glob.glob(pat) if os.path.exists(os.path.join(d,"DONE"))}
sets = {"v1": dict(main=arm("runs/seed*"), ctrl=arm("runs/ablate_seed*"), cheat=arm("runs/cheat_seed*")),
        "v2": dict(main=arm("runs_v2/*/runs/v2_normal_seed*"), ctrl=arm("runs_v2/*/runs/v2_ablate_seed*"),
                   cheat_stranger=arm("runs_v2/*/runs/v2_cheat_stranger_seed*"), cheat_evolve=arm("runs_v2/*/runs/v2_cheat_evolve_seed*"),
                   planted_honest=arm("runs_v2/*/runs/v2_planted_honest_seed*"))}
R = {}
for ver, S in sets.items():
    r = {}
    seeds = sorted(set(S["main"]) & set(S["ctrl"]))
    for label, allw, (g0, g1) in (("honesty_allworlds_g81_100", True, (81, 100)), ("honesty_startworld_g91_100", False, (91, 100)),
                                  ("promise_keeping_allworlds_g81_100", True, (81, 100))):
        k = "honesty" if "honesty" in label else "promise_keeping"
        m = [win(prog(S["main"][s]), g0, g1, k, allw) for s in seeds]; c = [win(prog(S["ctrl"][s]), g0, g1, k, allw) for s in seeds]
        r[label] = dict(main=ci(m), control=ci(c), diff=ci([a - b for a, b in zip(m, c)]), pairs_main_higher=sum(a > b for a, b in zip(m, c)), n_pairs=len(seeds))
    r["honesty_allworlds_g1_main"] = ci([win(prog(S["main"][s]), 1, 1, "honesty") for s in seeds])
    r["rep_honesty_corr_main"] = dict(g1_20=ci([win(prog(S["main"][s]), 1, 20, "rep_honesty_corr") for s in seeds]), g81_100=ci([win(prog(S["main"][s]), 81, 100, "rep_honesty_corr") for s in seeds]))
    for a in [k for k in S if k.startswith("cheat") or k == "planted_honest"]:
        rows = {s: {k: win(prog(d), 81, 100, k) for k in ("cheat_births", "honest_births", "cheat_survival", "cheat_bond_share", "cheat_propensity", "cheat_forced", "cheat_forced_on_partner", "honesty")} for s, d in S[a].items()}
        rows = {s: {k: v for k, v in x.items() if v is not None} for s, x in rows.items()}
        summ = {k: ci([x.get(k) for x in rows.values()]) for k in next(iter(rows.values()))}
        cb, hb = summ["cheat_births"]["mean"], summ["honest_births"]["mean"]
        summ["honest_to_cheat_offspring_ratio"] = round(hb / cb, 1) if cb else None
        r[a] = dict(n=len(rows), summary=summ)
    R[ver] = r
json.dump(R, open("runs_v2/v1_v2_compare.json", "w"), indent=1)
for ver in R:
    print("=====", ver)
    for k, v in R[ver].items():
        if isinstance(v, dict) and "summary" in v: print(k, v["n"], {kk: (vv["mean"], vv.get("ci95")) if isinstance(vv, dict) else vv for kk, vv in v["summary"].items()})
        else: print(k, json.dumps(v))
