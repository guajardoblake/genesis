"""Shared GENESIS demo driver (used by the Gradio app and the in-browser Pyodide page).
Drives the repo's own engine (Genesis.step_env / Surrogate.fit / Genesis.lineage) in-process; no forked logic."""
import os, sys, time, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if os.path.isdir(os.path.join(p, "genesis")): sys.path.insert(0, p); break
from genesis import engine as E, genome as GN

SHOW_AXES = ["love_wiring", "exploration", "memory_type"]

class LocalPool:  # the engine only needs .imap
    def imap(self, f, jobs): return map(f, jobs)

def _int(x, lo, hi, d):
    try:
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")): return d
        return int(min(hi, max(lo, round(v))))
    except (TypeError, ValueError): return d

def clamp_args(generations, population, seed):
    p = _int(population, 64, 256, 128)
    return dict(generations=_int(generations, 2, 40, 15), population=p if p in (64, 128, 256) else 128, seed=_int(seed, 0, 999999, 0))

def run(generations=20, population=256, seed=0, time_cap=75.0, on_gen=None):
    a = clamp_args(generations, population, seed); generations, population, seed = a["generations"], a["population"], a["seed"]
    out = tempfile.mkdtemp(prefix="genesis_demo_")
    g = E.Genesis(out, seed=int(seed), workers=1, batch=int(population), shard=max(16, int(population) // 4), screen_n=20_000)
    pool = LocalPool(); hist = []; t0 = time.time()
    for k in range(int(generations)):
        r = g.step_env(pool, 0)
        if len(g.sur.y): g.sur.fit()
        lin = g.lineage([r]); g.gen += 1
        o = r["o"] if r else None
        hist.append(dict(gen=g.gen, births=float(o["births"].mean()) if o is not None else 0.0,
                         law=float(o["law_breaks"].mean()) if o is not None else 0.0,
                         coop=float(o["coop"].mean()) if o is not None else 0.0,
                         niches=len(g.archives[0]), lin=lin))
        if on_gen: on_gen(k + 1, int(generations))
        if time.time() - t0 > time_cap: break
    last = next((h["lin"] for h in reversed(hist) if h["lin"] and SHOW_AXES[0] in h["lin"]), None)
    choices = {a: {"values": [str(v) for v in GN.AXES[GN.NAMES.index(a)][1]],
                   "share": [last[a].get(str(v), 0) for v in GN.AXES[GN.NAMES.index(a)][1]] if last else None} for a in SHOW_AXES}
    best = sorted(g.archives[0].values(), key=lambda r: -r["key"])[:1]
    top = None
    if best:
        b = best[0]; top = dict(recipe={k: str(v) for k, v in GN.describe(b["g"]).items()},
                                births=b["o"]["births"], law_breaks=b["o"]["law_breaks"])
    return dict(hist=[{k: v for k, v in h.items() if k != "lin"} for h in hist], choices=choices, top=top,
                stats={k: int(v) for k, v in g.stats.items()}, elapsed=time.time() - t0, niches=len(g.archives[0]))

def summary_md(res):
    h = res["hist"]; st = res["stats"]
    s = (f"Finished {len(h)} generations in {res['elapsed']:.1f} s. Simulated {st['sim_genomes']:,} genomes "
         f"({st['sim_agent_steps']:,} agent-steps); {res['niches']} archive niches filled. "
         f"Mean offspring went from {h[0]['births']:.3f} to {h[-1]['births']:.3f}. "
         "Tiny toy run with one seed: try another seed and expect different numbers.")
    if res["top"]:
        t = res["top"]; s += (f"\n\n**Top mind's recipe** (offspring {t['births']:.2f}, law-breaks/step {t['law_breaks']:.3f}):\n"
                              + "\n".join(f"- {k}: `{v}`" for k, v in t["recipe"].items()))
    return s
