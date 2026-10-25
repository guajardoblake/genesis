#!/usr/bin/env python3
"""GENESIS extra seeds on a free Kaggle TPU v5e-8 VM session (20 h/week quota, 9 h max per session).
The GENESIS engine is numpy (CPU) with an optional CuPy GPU path; the town loop is NOT ported to JAX, so the TPU chips
themselves are idle. What we use is the TPU VM's large host CPU: many run_box.py processes in parallel, exact v1 CPU
reference code (the path that produced the 10-seed result; it never had the GPU one-town bug). Before any seed runs,
the kernel reproduces 3 box-computed fingerprints (seed 0, batch 1024, 3 gens, main/cheat/ablate) bit-for-bit and
aborts on mismatch. New seeds 10..49 only (existing runs use 0..9; GPU batched uses seed 0)."""
import os, base64, io, tarfile, json
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(here)
REF = {"main": "7a324a22606a1bf6e0c538c05dcf2c41470d197237cbd99410481ba0ca178a0a",
       "cheat": "17e564bf24ebec5d6051f6da2bdad34b355e714d5d9d4e2f0784d1cb818fa26b",
       "ablate": "94dfada0ef41e189b1a0d105e81afffccdb7eab98932e81331de897450b66191"}
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    t.add("genesis", "genesis", filter=lambda x: None if "__pycache__" in x.name else x); t.add("run_box.py", "run_box.py")
b = base64.b64encode(buf.getvalue()).decode()
src = r'''# GENESIS tpu-seeds kernel (auto-built by kaggle/make_tpu_seeds.py). v1 rules, CPU numpy on the TPU VM host.
import base64, io, tarfile, os, subprocess, sys, time, json, glob, shutil, pickle, hashlib
T0 = time.time(); BUDGET = 8 * 3600 + 30 * 60      # stop cleanly 30 min before the 9 h TPU session cap
os.chdir("/kaggle/working"); tarfile.open(fileobj=io.BytesIO(base64.b64decode("@B64@")), mode="r:gz").extractall(".")
REF = @REF@
NC = os.cpu_count(); W = 8; P = max(1, (NC - 2) // W)
info = dict(cpus=NC, workers_per_run=W, parallel_runs=P)
try: info["mem_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
except Exception: pass
try:
    import jax; info["jax"] = jax.__version__; info["jax_devices"] = [str(d) for d in jax.devices()]
except Exception as e: info["jax_devices"] = "unavailable: " + repr(e)[:200]
print("machine", info, flush=True)
def fp(p):
    L = [json.loads(l) for l in open(p)]
    for d in L:
        for k in list(d):
            if "sec" in k or "time" in k or k == "elapsed_s": d.pop(k)
    return hashlib.sha256(json.dumps(L, sort_keys=True).encode()).hexdigest()
ARGS = {"main": [], "cheat": ["--cheaters", "0.1"], "ablate": ["--no-love", "--no-rep"]}
ver = {}
for n, a in ARGS.items():
    d = f"verify/{n}"; shutil.rmtree(d, ignore_errors=True)
    subprocess.run([sys.executable, "run_box.py", "--minutes", "20", "--max-gen", "3", "--seed", "0", "--workers", str(W), "--batch", "1024", "--out", d] + a, check=True, stdout=subprocess.DEVNULL)
    ver[n] = dict(got=fp(d + "/progress.jsonl"), want=REF[n]); ver[n]["match"] = ver[n]["got"] == ver[n]["want"]
info["verify"] = ver; info["verify_sec"] = round(time.time() - T0, 1)
json.dump(info, open("verify.json", "w"), indent=1); print("verify", ver, flush=True)
if not all(v["match"] for v in ver.values()):
    print("VERIFY FAILED: not running seeds", flush=True); sys.exit(1)
os.makedirs("runs", exist_ok=True)
for prev in glob.glob("/kaggle/input/*/runs"):          # resume from this kernel's previous output if attached
    for d in os.listdir(prev):
        s = os.path.join(prev, d)
        if not os.path.exists("runs/" + d): shutil.copytree(s, "runs/" + d) if os.path.isdir(s) else shutil.copy(s, "runs/" + d)
JOBS = []
for s in range(10, 50):
    JOBS += [(f"seed{s}", 176, ["--seed", str(s)]), (f"ablate_seed{s}", 100, ["--seed", str(s), "--no-love", "--no-rep"]),
             (f"cheat_seed{s}", 100, ["--seed", str(s), "--cheaters", "0.1"])]
JOBS = [j for j in JOBS if not os.path.exists(f"runs/{j[0]}/DONE")]
def gen_of(n):
    try: return pickle.load(open(f"runs/{n}/ckpt.pkl", "rb"))["gen"]
    except Exception: return 0
def sims(n):
    try: return json.loads(open(f"runs/{n}/progress.jsonl").readlines()[-1])["stats"]["sim_genomes"]
    except Exception: return 0
T1 = time.time(); sims0 = {}; live = {}; finished = []
def report():
    tot = sum(sims(n) - sims0.get(n, 0) for n in set(list(sims0) + finished + list(live)))
    el = time.time() - T1
    trios = sorted({int(n.rsplit("seed", 1)[1]) for n in finished} )
    full = [s for s in trios if all(os.path.exists(f"runs/{p}seed{s}/DONE") for p in ("", "ablate_", "cheat_"))]
    r = dict(info, seeds_runtime_h=round(el / 3600, 3), sim_genomes=tot, genomes_per_sec=round(tot / max(el, 1), 1),
             runs_done=sorted(finished), complete_seed_trios=full, seed_trios_per_hour=round(len(full) / max(el / 3600, 1e-9), 2),
             in_progress={n: gen_of(n) for n in live})
    json.dump(r, open("throughput.json", "w"), indent=1); return r
while (JOBS or live):
    left = BUDGET - (time.time() - T0)
    while JOBS and len(live) < P and left > 15 * 60:
        n, g, a = JOBS.pop(0); sims0[n] = sims(n)
        lf = open(f"runs/{n}.log", "a")
        live[n] = (subprocess.Popen([sys.executable, "run_box.py", "--minutes", str(int(left / 60) - 5), "--max-gen", str(g), "--workers", str(W), "--batch", "1024", "--out", f"runs/{n}"] + a, stdout=lf, stderr=subprocess.STDOUT), g)
    time.sleep(30)
    for n, (p, g) in list(live.items()):
        if p.poll() is not None:
            del live[n]
            if gen_of(n) >= g: open(f"runs/{n}/DONE", "w").close(); finished.append(n)
            print("ended", n, "gen", gen_of(n), "rc", p.returncode, flush=True)
    r = report()
    if left < 15 * 60 and not live: break
print("final", json.dumps(report()), flush=True)
'''.replace("@B64@", b).replace("@REF@", json.dumps(REF))
open("kaggle/k/tpu_seeds/job.py", "w").write(src)
json.dump({"id": "blakeguajardo/genesis-tpu-seeds", "title": "genesis-tpu-seeds", "code_file": "job.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": True, "enable_internet": False,
           "machine_shape": "TpuV5E8", "dataset_sources": [], "competition_sources": [], "kernel_sources": []},
          open("kaggle/k/tpu_seeds/kernel-metadata.json", "w"), indent=1)
print("built", len(src))
