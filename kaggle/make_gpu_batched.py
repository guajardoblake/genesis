#!/usr/bin/env python3
"""Build Kaggle GPU kernel blakeguajardo/genesis-gpu-batched: many independent 128-genome towns per GPU call
(town_xp town_size=128), two evolving populations side by side: reputation towns and control towns (no love, no rep),
both with 10% cheaters. Round-robin generations, checkpoint every gen to /kaggle/working, stop before Kaggle's 12h cap."""
import os, base64, io, tarfile, json
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(here)
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    t.add("genesis", "genesis", filter=lambda x: None if "__pycache__" in x.name else x)
b = base64.b64encode(buf.getvalue()).decode()
TOWNS = int(os.environ.get("TOWNS", 800)); HOURS = float(os.environ.get("HOURS", 10.5)); GENS = 100
src = f'''# GENESIS batched-towns GPU run (auto-built by kaggle/make_gpu_batched.py)
import base64, io, tarfile, os, sys, time, json, functools
T0 = time.time(); os.chdir("/kaggle/working")
tarfile.open(fileobj=io.BytesIO(base64.b64decode("{b}")), mode="r:gz").extractall("."); sys.path.insert(0, os.getcwd())
os.environ["GENESIS_TOWN"] = "128"
from genesis import engine, colab_jobs
gi = colab_jobs.gpu_info(); print("gpu", gi, flush=True)
import cupy as cp
engine._work = functools.partial(colab_jobs._gpu_work, xp=cp); engine.Pool = colab_jobs.SerialPool
cp.get_default_memory_pool().set_limit(fraction=0.85)
TOWNS, GENS, DEADLINE = {TOWNS}, {GENS}, T0 + {HOURS} * 3600
CFG = dict(rep=dict(cheaters=0.1), control=dict(cheaters=0.1, love=False, reputation=False))
E = {{}}
for n, ov in CFG.items():
    d = f"runs/{{n}}"
    E[n] = engine.Genesis.load(d) if os.path.exists(d + "/ckpt.pkl") else engine.Genesis(d, seed=0, batch=TOWNS * 128, shard=TOWNS * 128, workers=1, env_overrides=ov)
thr = dict(device=gi, towns_per_config=TOWNS, town_genomes=128, configs=list(CFG), gens_target=GENS, per_gen=[])
def save():
    s = [r for r in thr["per_gen"] if r["gen"] > 1]   # skip warm-up gen
    if s:
        sec = sum(r["sec"] for r in s); tg = sum(r["town_sims"] for r in s); gens = len(s)
        thr["town_generations_per_sec"] = round(TOWNS * gens / sec, 3)        # one town = 128 genomes, one generation of its population
        thr["town_sims_per_sec"] = round(tg / sec, 3)                          # every 128-genome town life simulated (all POET worlds/rungs)
        thr["sec_per_gen_at_towns"] = round(sec / gens, 2)
        # 100-seed experiment = 100 seeds x 1024 genomes (8 towns) x 100 gens = 80,000 town-generations
        thr["gpu_hours_per_100seed_experiment"] = round(80000 / thr["town_generations_per_sec"] / 3600, 2)
    thr["elapsed_h"] = round((time.time() - T0) / 3600, 3); thr["gens_done"] = {{n: g.gen for n, g in E.items()}}
    json.dump(thr, open("throughput.json", "w"), indent=1)
lastdt = 0
while min(g.gen for g in E.values()) < GENS:
    for n, g in E.items():
        if g.gen >= GENS: continue
        if time.time() + 1.5 * lastdt > DEADLINE: break
        s0 = g.stats["sim_genomes"]; t = time.time(); g.run(1e9, max_gen=g.gen + 1, log=lambda s: print(n, s[:250], flush=True))
        dt = time.time() - t; lastdt = max(lastdt * 0.5, dt)
        thr["per_gen"].append(dict(cfg=n, gen=g.gen, sec=round(dt, 2), town_sims=(g.stats["sim_genomes"] - s0) / 128)); save()
    else: continue
    break
save()
for n, g in E.items():
    if g.gen >= GENS: open(f"runs/{{n}}/DONE", "w").close()
print("finished", thr["gens_done"], "elapsed_h", thr["elapsed_h"], flush=True)
'''
d = "kaggle/k/gpu_batched"; os.makedirs(d, exist_ok=True); open(f"{d}/job.py", "w").write(src)
json.dump({"id": "blakeguajardo/genesis-gpu-batched", "title": "genesis-gpu-batched", "code_file": "job.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
           "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, open(f"{d}/kernel-metadata.json", "w"), indent=1)
print("built", d)
