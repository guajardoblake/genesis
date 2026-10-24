"""Two heavy free-Colab jobs for GENESIS. Both resume from a Drive folder and stop cleanly before the 12h limit.
  A  sweep : phase diagram, cheater fraction x reputation-signal noise x seeds (short evolutions, all cores)
  B  scale : 10x+ population evolution; GPU (CuPy) town sim when a GPU is present, else multicore numpy
Usage: python -m genesis.colab_jobs {sweep|scale} --drive DIR [--hours 11.5] [--tiny]"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
import sys, json, time, glob, shutil, argparse, pickle, functools, numpy as np

# ---------------- shared helpers ----------------
class SerialPool:  # stands in for multiprocessing.Pool inside worker processes / GPU mode
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def imap(self, f, it): return map(f, it)

def gpu_info():
    if os.environ.get("GENESIS_FAKE_GPU"):   # test hook: CuPy code path on CPU (tests/fakecupy.py)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")); import fakecupy
        sys.modules["cupy"] = fakecupy.cp; fakecupy.cp.get_default_memory_pool = lambda: type("M", (), {"set_limit": lambda self, **k: None})()
        return dict(name="fake-gpu(cpu)", free_gb=4.0, total_gb=4.0)
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        if n < 1: return None
        free, total = cp.cuda.runtime.memGetInfo(); name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        a = cp.arange(10) * 2; assert int(a.sum()) == 90
        return dict(name=name, free_gb=free / 1e9, total_gb=total / 1e9)
    except Exception as e:
        return None

def ram_gb():
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable"): return int(l.split()[1]) / 1e6
    except Exception: pass
    return 8.0

def log(path, msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg; print(line, flush=True)
    with open(path, "a") as f: f.write(line + "\n")

def summarize(rows, K=3):
    """Mean of the last K generations' law metrics (final-rung, env 0)."""
    tail = [r["law"][0] for r in rows[-K:] if r.get("law") and r["law"][0]]
    if not tail: return {}
    ks = ["honesty", "rep_honesty_corr", "cheat_frac", "cheat_survival", "cheat_births", "honest_births", "promise_keeping", "rep_gap_mean", "cheat_bond_share"]
    out = {}
    for k in ks:
        v = [t[k] for t in tail if t.get(k) is not None]
        out[k] = float(np.mean(v)) if v else None
    return out

# ---------------- Job A: phase-diagram sweep ----------------
def _cell_worker(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    cfrac, noise, seed, gens, batch, outdir = args
    from genesis import engine, town_xp
    engine.Pool = SerialPool; engine.run_town = town_xp.run_town
    ov = dict(rep_noise=noise)
    if cfrac > 0: ov["cheaters"] = cfrac
    tmp = os.path.join(outdir, "tmp", f"c{cfrac:.3f}_n{noise:.3f}_s{seed}")
    shutil.rmtree(tmp, ignore_errors=True)
    g = engine.Genesis(tmp, seed=seed, workers=1, batch=batch, screen_n=20000, env_overrides=ov)
    rows = []; t0 = time.time()
    g.run(1e9, max_gen=gens, log=lambda s: rows.append(json.loads(s)))
    res = dict(cheaters=cfrac, rep_noise=noise, seed=seed, gens=gens, batch=batch, secs=round(time.time() - t0, 1),
               **summarize(rows), last=rows[-1]["law"][0] if rows and rows[-1].get("law") else None)
    shutil.rmtree(tmp, ignore_errors=True)
    return res

def sweep(drive, hours, tiny=False, cores=None):
    from multiprocessing import Pool
    os.makedirs(drive, exist_ok=True); L = os.path.join(drive, "sweep.log"); resf = os.path.join(drive, "cells.jsonl")
    cores = cores or os.cpu_count(); t_end = time.time() + hours * 3600
    plan_f = os.path.join(drive, "plan.json")
    if os.path.exists(plan_f): plan = json.load(open(plan_f)); log(L, f"resuming plan {plan}")
    else:
        batch = 64 if tiny else 256
        # benchmark: one short cell on this machine (single core), then size the densest grid that fits
        tb = time.time(); _cell_worker((0.1, 0.1, 999, 2, batch, drive)); per_gen = (time.time() - tb) / 2
        ram_cores = max(1, int(ram_gb() / 1.2))           # ~1.2 GB peak per cell process, leaves margin
        workers = max(1, min(cores, ram_cores))
        budget = (hours * 3600 - (time.time() - tb) - min(600, 0.05 * hours * 3600)) * workers * 0.85   # 15% slack + 10 min margin
        best = None   # densest grid x most seeds x longest evolutions that fit (cost ~ linear in batch)
        for b in ([batch] if tiny else (512, 256)):
            for gens in ([3] if tiny else (60, 40, 30, 20, 15)):
                for grid in ([3] if tiny else (20, 16, 14, 12, 10, 8)):
                    for seeds in ([1] if tiny else (8, 6, 5, 4, 3)):
                        cost = grid * grid * seeds * gens * per_gen * (b / batch) * 2.0  # x2: cheater cells bigger + POET worlds grow
                        score = grid * grid * min(seeds, 6) * min(gens, 40) * (b / 256) ** 0.5
                        if cost <= budget and (best is None or score > best[0]): best = (score, grid, seeds, gens, b)
        if best is None: best = (0, 8, 3, 10, batch)
        _, grid, seeds, gens, batch = best
        plan = dict(grid=grid, seeds=seeds, gens=gens, batch=batch, workers=workers, per_gen_s=round(per_gen, 2),
                    cheaters=[round(x, 4) for x in np.linspace(0.0, 0.5, grid)], rep_noise=[round(x, 4) for x in np.linspace(0.0, 0.5, grid)],
                    est_hours=round(grid * grid * seeds * gens * per_gen * (batch / (64 if tiny else 256)) * 2.0 / workers / 3600, 2))
        json.dump(plan, open(plan_f, "w")); log(L, f"plan {plan}")
    done = set()
    if os.path.exists(resf):
        for l in open(resf):
            r = json.loads(l); done.add((r["cheaters"], r["rep_noise"], r["seed"]))
    todo = [(c, n, s, plan["gens"], plan["batch"], drive) for s in range(plan["seeds"]) for c in plan["cheaters"] for n in plan["rep_noise"] if (c, n, s) not in done]
    log(L, f"{len(done)} cells done, {len(todo)} to go, {plan['workers']} workers")
    with Pool(plan["workers"], maxtasksperchild=4) as pool:
        it = pool.imap_unordered(_cell_worker, todo)
        for i, r in enumerate(it):
            with open(resf, "a") as f: f.write(json.dumps(r) + "\n")
            if i % 10 == 0: log(L, f"cell {len(done)+i+1}/{len(done)+len(todo)}: c={r['cheaters']} n={r['rep_noise']} s={r['seed']} honesty={r.get('honesty')} {r['secs']}s")
            if time.time() > t_end:
                log(L, "time budget reached; stopping cleanly (re-run to resume)"); pool.terminate(); break
    render(drive)

def render(drive):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plan = json.load(open(os.path.join(drive, "plan.json"))); rows = [json.loads(l) for l in open(os.path.join(drive, "cells.jsonl"))] if os.path.exists(os.path.join(drive, "cells.jsonl")) else []
    C, N = plan["cheaters"], plan["rep_noise"]; mets = ["honesty", "rep_honesty_corr", "cheat_survival", "cheat_births"]
    grids = {m: np.full((len(C), len(N)), np.nan) for m in mets}; cnt = np.zeros((len(C), len(N)))
    acc = {m: [[[] for _ in N] for _ in C] for m in mets}
    for r in rows:
        i, j = C.index(r["cheaters"]), N.index(r["rep_noise"]); cnt[i, j] += 1
        for m in mets:
            if r.get(m) is not None: acc[m][i][j].append(r[m])
    for m in mets:
        for i in range(len(C)):
            for j in range(len(N)):
                if acc[m][i][j]: grids[m][i, j] = np.mean(acc[m][i][j])
    fig, ax = plt.subplots(1, 4, figsize=(22, 5))
    for a, m in zip(ax, mets):
        im = a.imshow(grids[m], origin="lower", aspect="auto", extent=[N[0], N[-1], C[0], C[-1]], cmap="viridis")
        a.set_title(m); a.set_xlabel("reputation-signal noise (sd)"); a.set_ylabel("cheater fraction"); fig.colorbar(im, ax=a)
    fig.suptitle(f"GENESIS phase diagram: {int(cnt.sum())} runs, {plan['grid']}x{plan['grid']} x {plan['seeds']} seeds x {plan['gens']} gens")
    fig.tight_layout(); fig.savefig(os.path.join(drive, "phase_diagram.png"), dpi=110); plt.close(fig)
    json.dump({m: grids[m].tolist() for m in mets} | dict(cheaters=C, rep_noise=N, counts=cnt.tolist()), open(os.path.join(drive, "phase_grid.json"), "w"))
    print("rendered", os.path.join(drive, "phase_diagram.png"), int(cnt.sum()), "runs")

# ---------------- Job B: scale-up ----------------
def _gpu_work(args, xp=None):
    from genesis.engine import rung_cfg
    from genesis import town_xp
    g, env, seed, rung, init_w = args
    return town_xp.run_town(g, env, seed, return_w=True, init_w=init_w, xp=xp, **rung_cfg(env, rung))

def scale(drive, hours, tiny=False, seed=0, cheaters=0.1):
    from genesis import engine, town_xp
    os.makedirs(drive, exist_ok=True); L = os.path.join(drive, "scale.log"); t_start = time.time()
    local = "/content/genesis_scale" if os.path.isdir("/content") else os.path.join(drive, "_local")
    os.makedirs(local, exist_ok=True)
    gi = gpu_info(); log(L, f"gpu={gi} cores={os.cpu_count()} ram_avail_gb={ram_gb():.1f}")
    if gi:
        import cupy as cp
        engine._work = functools.partial(_gpu_work, xp=cp); engine.Pool = SerialPool
        mempool = cp.get_default_memory_pool(); mempool.set_limit(fraction=0.85)
    else:
        engine.run_town = town_xp.run_town
    for f in ("ckpt.pkl", "progress.jsonl", "lineage.jsonl", "plan.json"):   # Drive -> local on resume
        if os.path.exists(os.path.join(drive, f)) and not os.path.exists(os.path.join(local, f)): shutil.copy(os.path.join(drive, f), local)
    plan_f = os.path.join(local, "plan.json"); ov = dict(cheaters=cheaters) if cheaters else {}
    if os.path.exists(plan_f): plan = json.load(open(plan_f)); log(L, f"resume plan {plan}")
    else:
        # benchmark: one generation at two sizes -> linear cost model -> biggest population with >= target gens
        def bench(batch, shard):
            d = os.path.join(local, f"_bench{batch}"); shutil.rmtree(d, ignore_errors=True)
            g = engine.Genesis(d, seed=seed, batch=batch, shard=shard, workers=(1 if gi else os.cpu_count()), env_overrides=ov)
            t = time.time(); g.run(1e9, max_gen=2, log=lambda s: None); dt = (time.time() - t) / 2; shutil.rmtree(d, ignore_errors=True); return dt
        b1, b2 = (64, 128) if tiny else (1024, 4096)
        sh = lambda b: (b if gi else 128)          # GPU: one big town per shard call; CPU: base town size
        t1, t2 = bench(b1, sh(b1)), bench(b2, sh(b2)); slope = max((t2 - t1) / (b2 - b1), 1e-6); icpt = max(t1 - slope * b1, 0.0)
        avail = hours * 3600 - (time.time() - t_start) - min(900, 0.1 * hours * 3600)   # 15 min margin for final save/sync
        target_gens = 3 if tiny else 100
        ENVF = 4.0   # POET grows up to 6 worlds; each gen evolves every world (measured ~3-5x the 2-gen benchmark)
        cands = [64, 128] if tiny else [10240, 16384, 20480, 32768, 40960, 65536]
        # memory cap: ~6 KB/mind state (float32), final rung 6 minds/genome; keep under 60% of free VRAM / RAM
        mem_gb = (gi["free_gb"] if gi else ram_gb()) * 0.6
        best = cands[0]
        for b in cands:
            mind_gb = b * 2 * 6e3 / 1e9 * 3        # rung-0 town: 2 minds/genome, x3 temporaries
            gens = avail / ((icpt + slope * b) * ENVF) * 0.85
            if gens >= target_gens and mind_gb < mem_gb: best = b
        gens = int(avail / ((icpt + slope * best) * ENVF) * 0.85)
        plan = dict(batch=best, shard=sh(best), gens=max(gens, 1), seed=seed, cheaters=cheaters, device=(gi["name"] if gi else "cpu"),
                    bench=dict(b1=b1, t1=round(t1, 2), b2=b2, t2=round(t2, 2)), est_hours=round(gens * (icpt + slope * best) * ENVF / 3600, 2),
                    baseline_batch=1024, scale_x=round(best / 1024, 1))
        json.dump(plan, open(plan_f, "w")); log(L, f"plan {plan}")
    ck = os.path.join(local, "ckpt.pkl")
    g = engine.Genesis.load(local) if os.path.exists(ck) else engine.Genesis(local, seed=plan["seed"], batch=plan["batch"], shard=plan["shard"], workers=(1 if gi else os.cpu_count()), env_overrides=ov)
    g.out = local
    if not gi: g.workers = os.cpu_count()
    log(L, f"start at gen {g.gen}/{plan['gens']}")
    last_sync = time.time(); SYNC = 60 if tiny else 1200
    def sync():
        for f in ("ckpt.pkl", "progress.jsonl", "lineage.jsonl", "plan.json"):
            p = os.path.join(local, f)
            if os.path.exists(p) and os.path.abspath(local) != os.path.abspath(drive):
                shutil.copy(p, os.path.join(drive, f + ".tmp")); os.replace(os.path.join(drive, f + ".tmp"), os.path.join(drive, f))
    t_end = t_start + hours * 3600 - min(600, 0.05 * hours * 3600)
    while g.gen < plan["gens"] and time.time() < t_end:
        g.run(min(SYNC, t_end - time.time()) / 60, max_gen=plan["gens"], log=lambda s: log(L, s[:300]))
        sync(); last_sync = time.time(); log(L, f"synced gen {g.gen} to Drive")
    if g.gen >= plan["gens"]: open(os.path.join(drive, "DONE"), "w").close()
    sync(); log(L, f"stopped at gen {g.gen}/{plan['gens']} (re-run to resume)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("job", choices=["sweep", "scale", "render"]); ap.add_argument("--drive", required=True)
    ap.add_argument("--hours", type=float, default=11.5); ap.add_argument("--tiny", action="store_true"); ap.add_argument("--cores", type=int, default=None)
    a = ap.parse_args()
    if a.job == "sweep": sweep(a.drive, a.hours, a.tiny, a.cores)
    elif a.job == "scale": scale(a.drive, a.hours, a.tiny)
    else: render(a.drive)
