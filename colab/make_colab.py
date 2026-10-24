#!/usr/bin/env python3
"""Build the two self-contained Colab notebooks (code embedded). Cell 1 runs everything."""
import os, io, base64, tarfile, json
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(here)
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    t.add("genesis", "genesis", filter=lambda x: None if "__pycache__" in x.name else x); t.add("run_box.py", "run_box.py")
B = base64.b64encode(buf.getvalue()).decode()
def nb(job, title, about):
    c1 = f'''# {title}
# {about}
# Run this ONE cell. It mounts Google Drive, resumes from MyDrive/GENESIS/{job} if a previous session saved work,
# auto-sizes to this machine (cores / RAM / GPU), stops cleanly at ~11.5 h, and saves a tar of the results.
import os, io, sys, base64, tarfile, subprocess, time, shutil
DRIVE_ROOT = "/content/drive/MyDrive/GENESIS"
try:
    from google.colab import drive; drive.mount("/content/drive"); os.makedirs(DRIVE_ROOT, exist_ok=True)
except Exception as e:
    print("Drive not mounted (", e, ") -> saving under /content/GENESIS (lost if the session ends)"); DRIVE_ROOT = "/content/GENESIS"
OUT = os.path.join(DRIVE_ROOT, "{job}"); os.makedirs(OUT, exist_ok=True)
CODE = "/content/genesis_code"; os.makedirs(CODE, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode("{B}")), mode="r:gz").extractall(CODE)
HOURS = float(os.environ.get("GENESIS_HOURS", "11.5")); TINY = os.environ.get("GENESIS_TINY") == "1"
cmd = [sys.executable, "-u", "-m", "genesis.colab_jobs", "{job}", "--drive", OUT, "--hours", str(HOURS)] + (["--tiny"] if TINY else [])
print("running:", " ".join(cmd), flush=True)
p = subprocess.Popen(cmd, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    if '"gen"' in line and len(line) > 400: line = line[:400] + "...\\n"
    print(line, end="", flush=True)
print("exit code", p.wait())
if "{job}" == "sweep": subprocess.run([sys.executable, "-m", "genesis.colab_jobs", "render", "--drive", OUT], cwd=CODE)
TAR = f"/content/genesis_{job}_results.tar.gz"
with tarfile.open(TAR, "w:gz") as t:
    for f in sorted(os.listdir(OUT)):
        if f in ("tmp", "_local", "ckpt.pkl") or f.endswith(".tmp"): continue  # ckpt stays on Drive (can be GBs)
        t.add(os.path.join(OUT, f), arcname="{job}/" + f)
shutil.copy(TAR, os.path.join(DRIVE_ROOT, os.path.basename(TAR)))
print("results tar:", TAR, os.path.getsize(TAR) // 1024, "KB (copy also in", DRIVE_ROOT, ")")
try:
    from IPython.display import Image, display
    if os.path.exists(os.path.join(OUT, "phase_diagram.png")): display(Image(os.path.join(OUT, "phase_diagram.png")))
    from google.colab import files; files.download(TAR)
except Exception as e: print("(download/display skipped:", e, ")")
'''
    c2 = f'''# Optional: small text-only copy of results (no checkpoint) as base64 chunks for copy/paste
import tarfile, io, base64, os
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    for f in sorted(os.listdir(OUT)):
        if f.endswith((".jsonl", ".json", ".log", ".png")) or f == "DONE": t.add(os.path.join(OUT, f), arcname="{job}/" + f)
s = base64.b64encode(buf.getvalue()).decode(); print("CHUNKS", (len(s) + 49999) // 50000)
for i in range(0, len(s), 50000): print(f"---CHUNK {{i // 50000}}---"); print(s[i:i + 50000])
'''
    return {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c} for c in (c1, c2)],
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "accelerator": "GPU" if job == "scale" else "None",
                         "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb("sweep", "GENESIS Job A (DTOM AI Labs account): phase-diagram sweep",
             "Grid: added-cheater fraction x reputation-signal noise x seeds; short evolutions in parallel on all cores. CPU runtime is enough."),
          open("colab/genesis_colab_dtom.ipynb", "w"))
json.dump(nb("scale", "GENESIS Job B (Blake personal account): 10x+ population scale-up",
             "Runtime > Change runtime type > T4 GPU. Uses CuPy on the GPU (falls back to all-core numpy if no GPU). Auto-sizes population to the hardware."),
          open("colab/genesis_colab_personal.ipynb", "w"))
print("wrote colab/genesis_colab_dtom.ipynb, colab/genesis_colab_personal.ipynb", len(B) // 1024, "KB code each")
