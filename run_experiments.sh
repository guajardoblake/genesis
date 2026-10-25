#!/bin/bash
# Queue after run_seeds.sh. Priority: paired main+control seeds 0-4, cheaters 0-4, then seeds 5-9, then long lives.
# Resume-safe (DONE markers + ckpt.pkl). Main = run-1 config to gen 176; control/cheater to gen 100 (compared to main at gen 100).
cd "$(dirname "$0")"
while pgrep -f run_seeds.sh >/dev/null; do sleep 60; done
job() { name=$1; gen=$2; shift 2; out=runs/$name; [ -f $out/DONE ] && return
  python3 run_box.py --minutes 600 --max-gen $gen --workers 7 --batch 1024 --out $out "$@" >> $out.log 2>&1 && \
  python3 -c "import pickle,sys;sys.exit(0 if pickle.load(open('$out/ckpt.pkl','rb'))['gen']>=$gen else 1)" && touch $out/DONE; }
for s in 0 1 2 3 4; do job seed$s 176 --seed $s; done
for s in 0 1 2 3 4; do job ablate_seed$s 100 --seed $s --no-love --no-rep; done
for s in 0 1 2 3 4; do job cheat_seed$s 100 --seed $s --cheaters 0.1; done
for s in 5 6 7 8 9; do job seed$s 176 --seed $s; job ablate_seed$s 100 --seed $s --no-love --no-rep; done
for s in 5 6 7 8 9; do job cheat_seed$s 100 --seed $s --cheaters 0.1; done
for s in 0 1; do job long_seed$s 60 --seed $s --long; done
echo ALL_DONE
