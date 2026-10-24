"""GENESIS coordinator: surrogate pre-screen -> successive-halving lives -> QD archive + novelty
-> POET world co-evolution -> lineage log. Checkpoints every generation."""
import os, json, time, pickle, numpy as np
from multiprocessing import Pool
from . import genome as GN
from .town import run_town, default_env
from .surrogate import Surrogate

RUNGS = [dict(T=32, per_genome=2), dict(T=96, per_genome=4), dict(T=192, per_genome=6)]  # Hyperband-style
KEEP = [0.34, 0.34]
BINS = 8  # descriptors: cooperation, closeness to partner, law-breaking rate
DESC = ("coop", "closeness", "law_breaks"); DLO = np.array([0, 0, 0]); DHI = np.array([0.6, 0.6, 0.15])

def law_metrics(o):
    """World-level law-keeping summary from one generation's final-rung outcomes (pooled over minds)."""
    from .town import promise_keeping_rate, reputation_gap
    if "promise_opps" not in o: return None
    gap = reputation_gap(o["rep_mean"], o["honesty"])
    pk = promise_keeping_rate(o["promise_breaks"], o["promise_opps"])
    hon = o["honesty"]; rp = o["rep_mean"]
    corr = float(np.corrcoef(rp, hon)[0, 1]) if len(hon) > 2 and hon.std() > 0 and rp.std() > 0 else None
    return dict(promise_keeping=None if pk != pk else round(pk, 5), promise_opps=round(float(o["promise_opps"].sum()), 2),
                rep_gap_mean=round(float(gap.mean()), 5), rep_gap_abs=round(float(np.abs(gap).mean()), 5),
                rep_mean=round(float(rp.mean()), 5), honesty=round(float(hon.mean()), 5),
                rep_honesty_corr=None if corr is None else round(corr, 4), n=int(len(hon)),
                **{k: round(float(o[k].mean()), 5) for k in o if k.startswith("cheat_")},
                honest_births=round(float(o["births"].mean()), 5))

def survives(o):  # minimal criterion = stayed alive and balanced (nature's filter, not a score)
    return (o["survival"] > 0.6) & (o["balanced"] > 0.3)

def repro_key(o):  # replacement in a niche: who left more offspring, ties broken by time alive
    return o["births"] + 0.01 * o["survival"]

def _work(args):
    g, env, seed, rung, init_w = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    rg = rung_cfg(env, rung)
    return run_town(g, env, seed, return_w=True, init_w=init_w, **rg)

def rung_cfg(env, rung):  # 'long' config scales every rung's life length (final rung 192 -> 512 at t_scale 8/3)
    rg = dict(RUNGS[rung]); rg["T"] = int(round(rg["T"] * env.get("t_scale", 1.0))); return rg

class Genesis:
    def __init__(self, out, seed=0, workers=None, lamarck=True, batch=1024, shard=128, screen_n=250_000, speed=True, env_overrides=None):
        self.out = out; os.makedirs(out, exist_ok=True); self.rng = np.random.default_rng(seed)
        self.workers = workers or max(1, os.cpu_count() - 1); self.lamarck = lamarck; self.batch = batch
        self.shard = shard; self.screen_n = screen_n; self.speed = speed
        self.envs = [dict(default_env(), **(env_overrides or {}))]; self.archives = [dict()]  # per env: cell -> record
        self.sur = Surrogate(); self.cache = {}; self.gen = 0; self.stats = dict(sim_genomes=0, sim_agent_steps=0, screened=0, cache_hits=0, early_dropped=0)
        self.novel = np.zeros((0, 3)); self.cell_gain = {}; self.t_sim = 0.0; self.t_total = 0.0

    # ---------- checkpoint ----------
    def save(self):
        tmp = os.path.join(self.out, "ckpt.tmp"); pool = None
        with open(tmp, "wb") as f: pickle.dump({k: v for k, v in self.__dict__.items() if k != "pool"}, f)
        os.replace(tmp, os.path.join(self.out, "ckpt.pkl"))
    @classmethod
    def load(cls, out):
        g = cls.__new__(cls)
        with open(os.path.join(out, "ckpt.pkl"), "rb") as f: g.__dict__.update(pickle.load(f))
        return g

    # ---------- helpers ----------
    def cell(self, o):
        d = np.stack([o[k] for k in DESC], 1); return np.clip(((d - DLO) / (DHI - DLO) * BINS).astype(int), 0, BINS - 1), d
    def evaluate(self, pool, g, env, rung, init_w=None):
        n = g.shape[0]; s = self.shard; jobs = []
        for i in range(0, n, s):
            iw = None
            if init_w is not None: iw = {k: v[i:i + s] for k, v in init_w.items()}
            jobs.append((g[i:i + s], env, int(self.rng.integers(1 << 31)), rung, iw))
        t = time.time(); res = list(pool.imap(_work, jobs))  # small shards -> idle workers steal the next one
        self.t_sim += time.time() - t
        out = {k: np.concatenate([r[k] for r in res]) for k in res[0]}
        self.stats["sim_genomes"] += n; rg = rung_cfg(env, rung); self.stats["sim_agent_steps"] += int(n * rg["T"] * rg["per_genome"])
        return out

    def propose(self, ei):
        """Adaptive scheduler: parents from niches that recently improved (or are new), plus mutation/crossover,
        plus surrogate-screened fresh genomes. Dedup against cache."""
        arc = self.archives[ei]; n = self.batch; parts = []; parent_w = []
        if arc:
            cells = list(arc); gains = np.array([self.cell_gain.get((ei, c), 1.0) for c in cells]) + 0.1
            births = np.array([arc[c]["o"]["births"] for c in cells]) + 0.05
            w = gains * births; w /= w.sum()
            pick = self.rng.choice(len(cells), int(n * 0.6), p=w)
            par = np.stack([arc[cells[i]]["g"] for i in pick])
            mate = np.stack([arc[cells[i]]["g"] for i in self.rng.choice(len(cells), len(pick), p=w)])
            kids = GN.mutate(self.rng, np.where(self.rng.random((len(pick), 1)) < 0.3, GN.crossover(self.rng, par, mate), par))
            parts.append(kids); parent_w += [cells[i] for i in pick]
        rest = n - sum(len(p) for p in parts)
        if self.speed and self.sur.w is not None:
            g, dt = self.sur.screen(self.rng, self.screen_n, rest); self.stats["screened"] += self.screen_n
        else:
            g = GN.sample(self.rng, rest)
        parts.append(g); parent_w += [None] * len(g)
        G = np.concatenate(parts)
        idx = GN.to_index(G); _, u = np.unique(idx, return_index=True); u = np.sort(u)
        fresh = np.array([((ei, int(idx[i])) not in self.cache) for i in u], bool) if self.speed else np.ones(len(u), bool)
        self.stats["cache_hits"] += int((~fresh).sum()) + (len(G) - len(u))
        u = u[fresh]
        init = None
        if self.lamarck and arc:
            m = np.array([parent_w[i] is not None for i in u])
            from .town import H, D_OBS, N_ACT
            Win = np.zeros((len(u), H, D_OBS), np.float32); Wout = np.zeros((len(u), N_ACT, H), np.float32)
            for j, i in enumerate(u):
                if parent_w[i] is not None and "Win" in arc[parent_w[i]]:
                    Win[j] = arc[parent_w[i]]["Win"]; Wout[j] = arc[parent_w[i]]["Wout"]
                else: m[j] = False
            init = dict(mask=m, Win=Win, Wout=Wout)
        return G[u], init

    def step_env(self, pool, ei):
        env = self.envs[ei]; g, init = self.propose(ei)
        if len(g) == 0: return {}
        o = None; alive_idx = np.arange(len(g))
        rungs = range(len(RUNGS)) if self.speed else [1]
        for r in rungs:
            iw = None if init is None else {k: v[alive_idx] for k, v in init.items()}
            o = self.evaluate(pool, g[alive_idx], env, r, iw)
            if r == rungs[0] and self.speed or not self.speed:
                y = o["births"] + o["survival"] * o["balanced"]; self.sur.add(g[alive_idx], y)
            for j, i in enumerate(alive_idx): self.cache[(ei, int(GN.to_index(g[i:i + 1])[0]))] = r
            if r == len(RUNGS) - 1 or not self.speed: break
            ok = survives(o)
            self.stats["early_dropped"] += int((~ok).sum())
            k = repro_key(o) + 0.5 * ok; keep = max(8, int(len(alive_idx) * KEEP[r]))
            sel = np.argsort(-k)[:keep]; alive_idx = alive_idx[sel]
        gg = g[alive_idx]; cells, d = self.cell(o)
        # novelty (kNN distance to behaviours already seen) breaks ties toward new behaviour
        if len(self.novel):
            sub = self.novel[self.rng.choice(len(self.novel), min(4000, len(self.novel)), replace=False)]
            nov = np.sort(np.linalg.norm(d[:, None] - sub[None], axis=2), 1)[:, :10].mean(1)
        else: nov = np.ones(len(d))
        self.novel = np.concatenate([self.novel, d])[-50000:]
        arc = self.archives[ei]; added = 0; mc = survives(o)
        for j in range(len(gg)):
            if not mc[j]: continue
            c = tuple(cells[j]); key = repro_key({k: v[j] for k, v in o.items()}) + 0.001 * nov[j]
            if c not in arc or key > arc[c]["key"]:
                gain = key - (arc[c]["key"] if c in arc else 0)
                arc[c] = dict(g=gg[j].copy(), key=float(key), o={k: float(v[j]) for k, v in o.items() if v.ndim == 1},
                              Win=o["Win"][j], Wout=o["Wout"][j], gen=self.gen)
                self.cell_gain[(ei, c)] = 0.7 * self.cell_gain.get((ei, c), 1.0) + gain; added += 1
        for c in arc: self.cell_gain[(ei, c)] = self.cell_gain.get((ei, c), 1.0) * 0.95
        return dict(o=o, g=gg, added=added, mc=int(mc.sum()), n0=len(g))

    def poet(self, pool):
        """Co-evolve worlds: an env whose minds pass the minimal criterion spawns a harder child.
        Child is kept only if transferred minds find it neither trivial nor impossible."""
        if self.gen % 3 != 2 or len(self.envs) >= 6: return None
        ei = int(self.rng.integers(len(self.envs))); arc = self.archives[ei]
        if len(arc) < 10: return None
        e = dict(self.envs[ei])
        e["food_rate"] *= self.rng.uniform(0.75, 0.95); e["temp_amp"] *= self.rng.uniform(1.0, 1.3)
        e["hazard"] *= self.rng.uniform(1.0, 1.5); e["cold_night"] *= self.rng.uniform(1.0, 1.3)
        best = sorted(arc.values(), key=lambda r: -r["key"])[:64]; g = np.stack([r["g"] for r in best])
        o = self.evaluate(pool, g, e, 1); frac = survives(o).mean()
        if 0.05 < frac < 0.9:
            self.envs.append(e); new = {}
            cells, _ = self.cell(o)
            for j in np.where(survives(o))[0]:
                c = tuple(cells[j]); k = float(repro_key({kk: v[j] for kk, v in o.items()}))
                if c not in new or k > new[c]["key"]:
                    new[c] = dict(g=g[j].copy(), key=k, o={kk: float(v[j]) for kk, v in o.items() if v.ndim == 1}, Win=o["Win"][j], Wout=o["Wout"][j], gen=self.gen)
            self.archives.append(new); return dict(spawned=e, transfer_pass=float(frac))
        return dict(rejected=e, transfer_pass=float(frac))

    def transfer(self, pool):
        """Minimal-criterion transfer: best minds from each env try the others; they move in if they qualify."""
        if len(self.envs) < 2 or self.gen % 4 != 3: return 0
        moved = 0
        for ei in range(len(self.envs)):
            others = [r for ej, a in enumerate(self.archives) if ej != ei for r in sorted(a.values(), key=lambda r: -r["key"])[:16]]
            if not others: continue
            g = np.stack([r["g"] for r in others]); o = self.evaluate(pool, g, self.envs[ei], 1); cells, _ = self.cell(o)
            for j in np.where(survives(o))[0]:
                c = tuple(cells[j]); k = float(repro_key({kk: v[j] for kk, v in o.items()}))
                if c not in self.archives[ei] or k > self.archives[ei][c]["key"]:
                    self.archives[ei][c] = dict(g=g[j].copy(), key=k, o={kk: float(v[j]) for kk, v in o.items() if v.ndim == 1}, Win=o["Win"][j], Wout=o["Wout"][j], gen=self.gen); moved += 1
        return moved

    def lineage(self, results):
        """Which design choices are carried by minds that actually reproduced (offspring-weighted)."""
        rows = []
        for r in results:
            if not r: continue
            w = r["o"]["births"] * survives(r["o"])
            rows.append((r["g"], w))
        if not rows: return None
        G = np.concatenate([a for a, _ in rows]); W = np.concatenate([b for _, b in rows])
        rec = {"gen": self.gen, "reproducers": int((W > 0).sum()), "offspring": float(W.sum())}
        if W.sum() > 0:
            for i, (name, vals) in enumerate(GN.AXES):
                h = np.bincount(G[:, i], weights=W, minlength=len(vals)) / W.sum()
                rec[name] = {str(v): round(float(x), 4) for v, x in zip(vals, h)}
        # also archive-wide (all envs) frequencies
        allg = np.stack([r["g"] for a in self.archives for r in a.values()]) if any(self.archives) else None
        if allg is not None:
            rec["archive"] = {name: {str(v): round(float(x), 4) for v, x in zip(vals, np.bincount(allg[:, i], minlength=len(vals)) / len(allg))} for i, (name, vals) in enumerate(GN.AXES)}
        return rec

    def run(self, minutes, log=print, max_gen=None):
        t_end = time.time() + minutes * 60; t0 = time.time()
        with Pool(self.workers) as pool:
            while time.time() < t_end and (max_gen is None or self.gen < max_gen):
                tg = time.time(); results = []
                for ei in range(len(self.envs)):
                    results.append(self.step_env(pool, ei))
                    if time.time() > t_end: break
                if len(self.sur.y) and (self.speed):
                    self.sur.fit()
                p = self.poet(pool); m = self.transfer(pool)
                lin = self.lineage(results)
                if lin:
                    with open(os.path.join(self.out, "lineage.jsonl"), "a") as f: f.write(json.dumps(lin) + "\n")
                self.gen += 1; self.t_total += time.time() - tg
                mcs = sum(r.get("mc", 0) for r in results if r)
                births = [float(r["o"]["births"].mean()) for r in results if r]
                line = dict(gen=self.gen, envs=len(self.envs), archive=[len(a) for a in self.archives], mc_pass=mcs,
                            mean_births_final_rung=[round(b, 3) for b in births], poet=p, transferred=m,
                            law=[law_metrics(r["o"]) for r in results if r],
                            stats=dict(self.stats), elapsed_s=round(time.time() - t0, 1))
                log(json.dumps(line, default=float))
                with open(os.path.join(self.out, "progress.jsonl"), "a") as f: f.write(json.dumps(line, default=float) + "\n")
                self.save()
