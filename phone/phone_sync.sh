#!/bin/bash
# Box side GENESIS phone sync (loop every 45s; `once` for a single pass).
#  - relaunches phone guard3 if its heartbeat is stale (Samsung restarts adbd on screen lock/unlock,
#    which kills every shell-uid process; jobs resume from their per-generation checkpoint)
#  - pulls progress/lineage/logs (never ckpt) into runs/<job>/
#  - keeps MAXQ unstarted jobs queued on the phone, claiming them atomically in runs/PHONE_OWNED
cd /workspace/genesis-export
PL=/workspace/phonelink; G=/data/local/tmp/gx; LOG=runs/phone_sync.log; MAXQ=${MAXQ:-1}
export PATH=/home/box/.local/bin:$PATH
lg(){ echo "$(date '+%F %T') $*" >> $LOG; }
priv(){ ( cd $PL && T=${T:-40} ./P "$1" ); }
claim(){ python3 - <<'PY'
import os, fcntl
ns = {}; exec(open("run_parallel.py").read().split("def live")[0], ns)
with open("runs/.phone_owned.lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    own = open("runs/PHONE_OWNED").read().split() if os.path.exists("runs/PHONE_OWNED") else []
    for n, g, a in ns["J"]:
        if n in own or os.path.exists(f"runs/{n}"): continue
        open("runs/PHONE_OWNED.tmp", "w").write("\n".join(own + [n]) + "\n"); os.replace("runs/PHONE_OWNED.tmp", "runs/PHONE_OWNED")
        print(f"{n}:{g}:{a}"); break
PY
}
once(){
  timeout 20 phone status >/dev/null 2>&1 || { lg "link down"; return; }
  exec 9>runs/.phone_sync.lock; flock -n 9 || { lg "another sync running, skip"; return; }
  ST=$(priv 'G=/data/local/tmp/gx; echo NOW=$(date +%s) HB=$(stat -c %Y $G/q/status.json 2>/dev/null || echo 0); cat $G/q/status.json; echo; cat $G/q/queue.txt; echo ===; for d in $G/runs/*/; do n=$(basename $d); g=$(tail -1 $d/progress.jsonl 2>/dev/null | grep -o "\"gen\": [0-9]*" | grep -o "[0-9]*$"); echo $n ${g:-0} $( [ -f $d/DONE ] && echo DONE); done')
  echo "$ST" | grep -q '^NOW=' || { lg "priv call failed"; return; }
  echo "$ST" > runs/phone_status.txt
  NOW=$(echo "$ST" | sed -n 's/^NOW=\([0-9]*\) HB=\([0-9]*\).*/\1/p'); HB=$(echo "$ST" | sed -n 's/^NOW=\([0-9]*\) HB=\([0-9]*\).*/\2/p')
  if [ $((NOW-HB)) -gt 90 ] && [ ! -f runs/PHONE_PAUSE ]; then
    T=12 priv 'G=/data/local/tmp/gx; [ -f $G/q/STOP ] || { cd $G; setsid sh $G/guard3.sh < /dev/null > $G/q/g3.out 2>&1 & sleep 3; echo relaunched; }' >/dev/null
    lg "guard heartbeat stale $((NOW-HB))s -> relaunched"
  fi
  priv 'G=/data/local/tmp/gx; rm -rf /sdcard/gxq/out; mkdir -p /sdcard/gxq/out; for d in $G/runs/*/; do n=$(basename $d); mkdir -p /sdcard/gxq/out/$n; cp $d/progress.jsonl $d/lineage.jsonl /sdcard/gxq/out/$n/ 2>/dev/null; [ -f $d/DONE ] && cp $d/DONE /sdcard/gxq/out/$n/; cp $G/runs/$n.log /sdcard/gxq/out/$n/run.log 2>/dev/null; done; cp $G/q/guard3.log /sdcard/gxq/out/; echo ok' >/dev/null
  rm -rf runs/phone_pull; mkdir -p runs/phone_pull
  if timeout 400 scp -q -r -S /home/box/.local/bin/ssh -F $PL/ssh_config phone:/sdcard/gxq/out runs/phone_pull/ 2>>$LOG; then
    for d in runs/phone_pull/out/*/; do n=$(basename $d); [ "$n" = phone_s5_cheat10 ] && continue
      mkdir -p runs/$n; cp -p $d/* runs/$n/; done; cp runs/phone_pull/out/guard3.log runs/phone_guard3.log 2>/dev/null
  else lg "pull failed"; fi
  Q=$(echo "$ST" | sed -n '3,/^===$/p' | cut -d: -f1)
  echo "$Q" > /tmp/phone_q.txt
  # reconcile: every owned unfinished job must be in the phone queue (claims are never released automatically)
  Q=$(echo "$ST" | sed -n '3,/^===$/p' | cut -d: -f1)
  python3 - <<'PY' > /tmp/phone_missing.txt
import os
ns = {}; exec(open("run_parallel.py").read().split("def live")[0], ns)
own=open("runs/PHONE_OWNED").read().split(); q=open("/tmp/phone_q.txt").read().split()
for n,g,a in ns["J"]:
    if n in own and n not in q and not os.path.exists(f"runs/{n}/DONE") and n!="cheat_seed5": print(f"{n}:{g}:{a}")
PY
  while read -r M; do [ -n "$M" ] && priv "grep -q '^${M%%:*}:' $G/q/queue.txt || echo '$M' >> $G/q/queue.txt; echo ok" >/dev/null && lg "re-pushed $M"; done < /tmp/phone_missing.txt
  PEND=$(echo "$ST" | python3 -c "
import sys,re; s=sys.stdin.read().split('\n'); i=s.index('===') if '===' in s else len(s)
run=set(re.findall(r'(\w+)@', s[1] if len(s)>1 else '')); q=[l.split(':')[0] for l in s[2:i] if ':' in l]
done={l.split()[0] for l in s[i+1:] if l.endswith('DONE')}
print(sum(1 for n in q if n not in run and n not in done))")
  while [ -n "$PEND" ] && [ "$PEND" -lt $MAXQ ]; do
    J=$(claim); [ -z "$J" ] && break
    priv "echo '$J' >> $G/q/queue.txt; echo ok" | grep -q ok && lg "claimed $J" || lg "claimed $J (push unconfirmed; reconcile next pass)"; PEND=$((PEND+1))
  done
  lg "sync ok: $(echo "$ST" | sed -n 2p)"
}
[ "$1" = once ] && { once; exit; }
while true; do once; sleep 45; done
