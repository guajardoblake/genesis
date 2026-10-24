#!/bin/bash
# Queue after run_seeds.sh: ablation, cheater and long-life configs. Resume-safe (DONE markers + ckpt).
cd "$(dirname "$0")"
while pgrep -f run_seeds.sh >/dev/null; do sleep 60; done
job() { name=$1; gen=$2; shift 2; out=runs/$name; [ -f $out/DONE ] && return
  python3 run_box.py --minutes 600 --max-gen $gen --workers 7 --batch 1024 --out $out "$@" >> $out.log 2>&1 && \
  python3 -c "import pickle,sys;sys.exit(0 if pickle.load(open('$out/ckpt.pkl','rb'))['gen']>=$gen else 1)" && touch $out/DONE; }
for s in 0 1 2; do job ablate_seed$s 100 --seed $s --no-love --no-rep; done
for s in 0 1 2; do job cheat_seed$s 100 --seed $s --cheaters 0.1; done
for s in 0 1; do job long_seed$s 60 --seed $s --long; done
echo ALL_DONE
