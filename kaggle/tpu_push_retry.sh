#!/bin/bash
# Retry pushing genesis-tpu-seeds every 10 min until Kaggle accepts it (blocked by 5-batch-session cap while sweep + v2-a..d run). Gives up after 14 h.
cd /workspace/genesis-export/kaggle/k/tpu_seeds
for i in $(seq 1 84); do
  out=$(kaggle kernels push -p . 2>&1 | grep -v outdated); echo "$(date '+%F %T %Z') $out"
  echo "$out" | grep -qi "error" || { echo PUSHED; exit 0; }
  sleep 600
done; echo GAVE_UP
