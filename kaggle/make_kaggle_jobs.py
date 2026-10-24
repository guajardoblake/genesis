#!/usr/bin/env python3
"""Build one Kaggle CPU kernel per GENESIS job (same flags as run_parallel.py J). Output: /kaggle/working/runs/<name>/ (same layout as box)."""
import os, sys, base64, io, tarfile, json
here=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(here)
ns={}; exec(open("run_parallel.py").read().split("def live")[0], ns); J={n:(g,a) for n,g,a in ns["J"]}
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode="w:gz") as t:
    t.add("genesis","genesis",filter=lambda x: None if "__pycache__" in x.name else x); t.add("run_box.py","run_box.py")
b=base64.b64encode(buf.getvalue()).decode()
for n in sys.argv[1:]:
    g,a=J[n]; slug="genesis-"+n.replace("_","-")
    src=f'''# GENESIS job {n} (auto-built by kaggle/make_kaggle_jobs.py). CPU numpy, same flags as box run_parallel.py.
import base64, io, tarfile, os, subprocess, sys, shlex
os.chdir("/kaggle/working"); tarfile.open(fileobj=io.BytesIO(base64.b64decode("{b}")),mode="r:gz").extractall(".")
os.makedirs("runs/{n}",exist_ok=True)
cmd=[sys.executable,"run_box.py","--minutes","680","--max-gen","{g}","--workers","3","--batch","1024","--out","runs/{n}"]+shlex.split("{a}")
with open("runs/{n}.log","a") as lf: r=subprocess.run(cmd,stdout=lf,stderr=subprocess.STDOUT)
sys.path.insert(0, os.getcwd())  # ckpt.pkl pickles genesis.* objects; unpickling needs the extracted package importable
import pickle
try: gen=pickle.load(open("runs/{n}/ckpt.pkl","rb"))["gen"]
except Exception as e:
    import json as _j; gen=max((_j.loads(l)["gen"] for l in open("runs/{n}/progress.jsonl")), default=0); print("ckpt read failed, using progress.jsonl:",e)
if gen>={g}: open("runs/{n}/DONE","w").close()
print("{n} finished gen",gen,"rc",r.returncode)
'''
    d=f"kaggle/k/{n}"; os.makedirs(d,exist_ok=True); open(f"{d}/job.py","w").write(src)
    json.dump({"id":f"blakeguajardo/{slug}","title":slug,"code_file":"job.py","language":"python","kernel_type":"script",
      "is_private":True,"enable_gpu":False,"enable_tpu":False,"enable_internet":False,"dataset_sources":[],"competition_sources":[],"kernel_sources":[]},open(f"{d}/kernel-metadata.json","w"),indent=1)
    print("built",d,slug)
