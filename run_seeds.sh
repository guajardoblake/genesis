#!/bin/bash
# Multi-seed replication of run 1 (same config: batch 1024, 7 workers, lamarck on, speed on), stop at gen 176.
# Resume-safe: re-running this script continues each seed from runs/seedN/ckpt.pkl.
cd "$(dirname "$0")"
for s in ${SEEDS:-0 1 2 3}; do
  out=runs/seed$s
  if [ -f $out/DONE ]; then continue; fi
  python3 run_box.py --minutes 600 --max-gen 176 --seed $s --workers 7 --batch 1024 --out $out >> $out.log 2>&1 && \
    python3 -c "import pickle,sys;d=pickle.load(open('$out/ckpt.pkl','rb'));sys.exit(0 if d['gen']>=176 else 1)" && touch $out/DONE
done
echo ALL_DONE
