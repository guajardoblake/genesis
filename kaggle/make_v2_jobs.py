#!/usr/bin/env python3
"""GENESIS v2 reruns on free Kaggle CPU. 4 kernels, each runs its queue sequentially (run_box, 4 workers),
time-boxed to stop cleanly before Kaggle's 12 h limit so /kaggle/working output is always saved.
Every run checkpoints each generation (ckpt.pkl); a re-push with kernel_sources=[own slug] resumes unfinished runs."""
import os, sys, base64, io, tarfile, json
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(here)
A = dict(normal="--rules v2", ablate="--rules v2 --no-love --no-rep",
         stranger="--rules v2 --cheaters 0.1 --cheat-mode stranger", evolve="--rules v2 --cheaters 0.1 --cheat-mode evolve",
         planted="--rules v2 --cheaters 0.1 --cheat-mode planted_honest")
NAME = dict(normal="v2_normal", ablate="v2_ablate", stranger="v2_cheat_stranger", evolve="v2_cheat_evolve", planted="v2_planted_honest")
def q(*pairs): return [(f"{NAME[a]}_seed{s}", 100, f"--seed {s} " + A[a]) for a, s in pairs]
K = {
 "genesis-v2-a": q(*[(a, s) for s in range(5) for a in ("normal", "ablate")], ("stranger", 0), ("stranger", 1)),
 "genesis-v2-b": q(*[(a, s) for s in range(5, 10) for a in ("normal", "ablate")], ("stranger", 2), ("stranger", 3)),
 "genesis-v2-c": q(*[("stranger", s) for s in range(4, 8)], *[("evolve", s) for s in range(4)], *[("planted", s) for s in range(5)]),
 "genesis-v2-d": q(*[("stranger", s) for s in (8, 9)], *[("evolve", s) for s in range(4, 10)], *[("planted", s) for s in range(5, 10)]),
}
assert sorted(n for v in K.values() for n, _, _ in v) == sorted(f"{NAME[a]}_seed{s}" for a in A for s in range(10))
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    t.add("genesis", "genesis", filter=lambda x: None if "__pycache__" in x.name else x); t.add("run_box.py", "run_box.py")
b = base64.b64encode(buf.getvalue()).decode()
for slug, jobs in K.items():
    src = f'''# GENESIS v2 rerun kernel {slug} (auto-built by kaggle/make_v2_jobs.py). CPU numpy, rules v2.
import base64, io, tarfile, os, subprocess, sys, shlex, time, json, glob, shutil, pickle
T0 = time.time(); BUDGET_MIN = 11 * 60 + 15      # stop cleanly ~45 min before Kaggle's 12 h limit
os.chdir("/kaggle/working"); tarfile.open(fileobj=io.BytesIO(base64.b64decode("{b}")), mode="r:gz").extractall(".")
sys.path.insert(0, os.getcwd())
JOBS = {json.dumps(jobs)}
os.makedirs("runs", exist_ok=True)
for prev in glob.glob("/kaggle/input/*/runs"):        # resume: previous output of this kernel attached as input
    for d in os.listdir(prev):
        if not os.path.exists("runs/" + d): shutil.copytree(os.path.join(prev, d), "runs/" + d) if os.path.isdir(os.path.join(prev, d)) else shutil.copy(os.path.join(prev, d), "runs/" + d)
def gen_of(n):
    try: return pickle.load(open(f"runs/{{n}}/ckpt.pkl", "rb"))["gen"]
    except Exception:
        try: return max(json.loads(l)["gen"] for l in open(f"runs/{{n}}/progress.jsonl"))
        except Exception: return 0
status = {{}}
for n, g, a in JOBS:
    if os.path.exists(f"runs/{{n}}/DONE"): status[n] = "done(prev)"; continue
    left = BUDGET_MIN - (time.time() - T0) / 60
    if left < 10: status[n] = f"not started, gen {{gen_of(n)}}"; continue
    os.makedirs(f"runs/{{n}}", exist_ok=True); t = time.time()
    cmd = [sys.executable, "run_box.py", "--minutes", str(round(left - 5, 1)), "--max-gen", str(g), "--workers", "4", "--batch", "1024", "--out", f"runs/{{n}}"] + shlex.split(a)
    with open(f"runs/{{n}}.log", "a") as lf: r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    k = gen_of(n)
    if k >= g: open(f"runs/{{n}}/DONE", "w").close()
    status[n] = f"gen {{k}}/{{g}} rc {{r.returncode}} {{(time.time() - t) / 60:.1f}} min"
    print(n, status[n], flush=True)
    json.dump(dict(status=status, elapsed_min=round((time.time() - T0) / 60, 1)), open("runs/STATUS.json", "w"), indent=1)
json.dump(dict(status=status, elapsed_min=round((time.time() - T0) / 60, 1)), open("runs/STATUS.json", "w"), indent=1)
for f in ("genesis", "run_box.py"): shutil.rmtree(f, ignore_errors=True) if os.path.isdir(f) else os.remove(f)
print("ALL", status)
'''
    d = f"kaggle/k/{slug}"; os.makedirs(d, exist_ok=True); open(f"{d}/job.py", "w").write(src)
    json.dump({"id": f"blakeguajardo/{slug}", "title": slug, "code_file": "job.py", "language": "python", "kernel_type": "script",
               "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
               "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, open(f"{d}/kernel-metadata.json", "w"), indent=1)
    print("built", d, len(jobs), "runs")
