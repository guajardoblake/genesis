#!/usr/bin/env python3
"""Multi-seed statistics: means with 95% t-intervals, paired main-vs-control differences, cheater test.
Usage: python3 multiseed_report.py  (reads runs/seed*, runs/ablate_seed*, runs/cheat_seed*, runs/long_seed*) -> runs/multiseed.json"""
import os, json, glob, numpy as np
try:
    from scipy.stats import t as _t
    def t95(df): return float(_t.ppf(0.975, df))
except ImportError:  # Cornish-Fisher inverse-t approximation (error < 0.001 for df >= 3)
    _T = {1: 12.706, 2: 4.303}
    def t95(df):
        if df in _T: return _T[df]
        z = 1.959964; z3, z5, z7 = z**3, z**5, z**7
        return z + (z3 + z)/(4*df) + (5*z5 + 16*z3 + 3*z)/(96*df**2) + (3*z7 + 19*z5 + 17*z3 - 15*z)/(384*df**3)

def ci(x):
    x = np.array([v for v in x if v is not None], float); n = len(x)
    if n == 0: return None
    m = float(x.mean()); h = float(t95(n - 1) * x.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    return dict(n=n, mean=round(m, 4), ci95=None if h is None else [round(m - h, 4), round(m + h, 4)], min=round(float(x.min()), 4), max=round(float(x.max()), 4))

def prog(d):
    P = {}
    for l in open(os.path.join(d, "progress.jsonl")): p = json.loads(l); P[p["gen"]] = p
    return P

def window(P, g0, g1, f):
    v = [f(P[g]) for g in range(g0, g1 + 1) if g in P]; v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None

def base_law(k): return lambda p: (p.get("law") or [None])[0] and p["law"][0].get(k)
def births0(p): return p["mean_births_final_rung"][0] if p["mean_births_final_rung"] else None

def lineage_lead(d, axis):
    L = [json.loads(l) for l in open(os.path.join(d, "lineage.jsonl"))]; L = {r["gen"]: r for r in L}.values()
    lead = [max(r[axis], key=r[axis].get) for r in L if axis in r]
    return max(set(lead), key=lead.count) if lead else None, (lead.count(max(set(lead), key=lead.count)) / len(lead) if lead else None)

def done(pat): return sorted(d for d in glob.glob(pat) if os.path.exists(os.path.join(d, "DONE")))
def seed(d): return int(d.rsplit("seed", 1)[1])

R = {}
main = {seed(d): d for d in done("runs/seed*")}
abl = {seed(d): d for d in done("runs/ablate_seed*")}
che = {seed(d): d for d in done("runs/cheat_seed*")}
# main config, final (gen 176)
rows = {}
for s, d in main.items():
    P = prog(d); last = P[max(P)]
    rows[s] = dict(births_g176=births0(last), births_last10=window(P, 167, 176, births0), worlds=last["envs"], niches=sum(last["archive"]),
                   genomes=last["stats"]["sim_genomes"],
                   promise_keeping_last20=window(P, 157, 176, base_law("promise_keeping")), honesty_last20=window(P, 157, 176, base_law("honesty")),
                   rep_gap_last20=window(P, 157, 176, base_law("rep_gap_mean")), rep_honesty_corr_last20=window(P, 157, 176, base_law("rep_honesty_corr")),
                   promise_keeping_first20=window(P, 1, 20, base_law("promise_keeping")), honesty_first20=window(P, 1, 20, base_law("honesty")),
                   love_lead=lineage_lead(d, "love_wiring"))
R["main_per_seed"] = rows
R["main_summary"] = {k: ci([r[k] for r in rows.values()]) for k in rows[next(iter(rows))] if k != "love_lead"} if rows else {}
# paired main vs control at gen 91-100
pair = {}
for s in sorted(set(main) & set(abl)):
    Pm, Pa = prog(main[s]), prog(abl[s])
    f = lambda P: dict(births=window(P, 91, 100, births0), promise_keeping=window(P, 91, 100, base_law("promise_keeping")),
                       honesty=window(P, 91, 100, base_law("honesty")), niches=sum(P[100]["archive"]), worlds=P[100]["envs"])
    pair[s] = dict(main=f(Pm), control=f(Pa))
R["paired_per_seed"] = pair
if pair:
    R["paired_summary"] = {k: dict(main=ci([p["main"][k] for p in pair.values()]), control=ci([p["control"][k] for p in pair.values()]),
                                   diff_main_minus_control=ci([None if p["main"][k] is None or p["control"][k] is None else p["main"][k] - p["control"][k] for p in pair.values()]))
                           for k in pair[next(iter(pair))]["main"]}
# cheater test
ch = {}
for s, d in che.items():
    P = prog(d)
    def cw(g0, g1, k): return window(P, g0, g1, lambda p: (p.get("law") or [None])[0] and p["law"][0].get(k))
    ch[s] = dict(cheat_frac=cw(1, 100, "cheat_frac"),
                 cheat_bond_share_first20=cw(1, 20, "cheat_bond_share"), cheat_bond_share_last20=cw(81, 100, "cheat_bond_share"),
                 cheat_births_last20=cw(81, 100, "cheat_births"), honest_births_last20=cw(81, 100, "honest_births"),
                 cheat_bonds_lost_last20=cw(81, 100, "cheat_bonds_lost"))
R["cheater_per_seed"] = ch
if ch: R["cheater_summary"] = {k: ci([r[k] for r in ch.values()]) for k in ch[next(iter(ch))]}
if ch: R["cheater_summary"]["bond_share_change_last_minus_first"] = ci([r["cheat_bond_share_last20"] - r["cheat_bond_share_first20"] for r in ch.values()])
lg = {}
for d in done("runs/long_seed*"):
    P = prog(d); lg[seed(d)] = dict(gens=max(P), births_last10=window(P, max(P) - 9, max(P), births0),
                                    promise_keeping_last10=window(P, max(P) - 9, max(P), base_law("promise_keeping")), worlds=P[max(P)]["envs"])
R["long_per_seed"] = lg
json.dump(R, open("runs/multiseed.json", "w"), indent=1); print(json.dumps(R, indent=1))
