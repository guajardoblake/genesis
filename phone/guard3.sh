#!/system/bin/sh
# GENESIS phone queue v3. Jobs from q/queue.txt (NAME:GEN:ARGS), slots/workers from q/conf.
# Thermal (battery sensor): >=44C halve workers, >=47C pause, resume <=43C.
# Battery: not charging & <=15% -> stop all (ckpt each gen, resumes). New/resumed job needs >15% (Blake 10-24: run on battery to 15%).
G=/data/local/tmp/gx; Q=$G/q; L=$G/python/lib; LOG=$Q/guard3.log; cd $G
mkdir -p $Q/run; touch $Q/queue.txt
PY="$L/ld-linux-aarch64.so.1 --library-path $L $G/python/bin/python3.11"
lg(){ echo "$(date +%T) $*" >> $LOG; }
alive(){ [ -f $Q/run/$1.pid ] && kill -0 $(cat $Q/run/$1.pid) 2>/dev/null; }
curgen(){ tail -1 $G/runs/$1/progress.jsonl 2>/dev/null | grep -o '"gen": [0-9]*' | grep -o '[0-9]*$' || echo 0; }
start(){ # name gen args workers
  mkdir -p $G/runs/$1
  setsid nice -n 5 $PY $G/run_phone2.py $3 --max-gen $2 --workers $4 --out $G/runs/$1 >> $G/runs/$1.log 2>&1 &
  echo $! > $Q/run/$1.pid; echo $4 > $Q/run/$1.w; lg "start $1 workers=$4 pid=$! gen=$(curgen $1)"; }
stopj(){ P=$(cat $Q/run/$1.pid 2>/dev/null); [ -n "$P" ] && { kill -CONT -$P 2>/dev/null; kill -TERM -$P 2>/dev/null; sleep 4; kill -KILL -$P 2>/dev/null; }; rm -f $Q/run/$1.pid; lg "stop $1"; }
PAUSED=0; HOT=0
lg "guard3 up pid=$$"; echo $$ > $Q/guard3.pid
while true; do
  [ -f $Q/STOP ] && { for f in $Q/run/*.pid; do [ -f "$f" ] && stopj $(basename $f .pid); done; lg "STOP file, exit"; exit 0; }
  SLOTS=2; WORKERS=4; . $Q/conf 2>/dev/null
  B=$(dumpsys battery); T=$(echo "$B"|sed -n 's/^ *temperature: //p'|head -1); LV=$(echo "$B"|sed -n 's/^ *level: //p'|head -1)
  CH=0; echo "$B" | grep -qE "(AC|USB|Wireless) powered: true" && CH=1
  W=$WORKERS; [ $HOT = 1 ] && W=$(( (WORKERS+1)/2 ))
  RUN=""; NR=0
  while IFS=: read N GN A; do [ -z "$N" ] && continue
    if alive $N; then
      NR=$((NR+1)); RUN="$RUN $N@$(curgen $N)"
      # worker count changed (config or thermal) -> restart from ckpt
      [ $PAUSED = 0 ] && [ "$(cat $Q/run/$N.w)" != "$W" ] && { stopj $N; start $N $GN "$A" $W; }
    elif [ -f $Q/run/$N.pid ]; then
      rm -f $Q/run/$N.pid
      if [ $(curgen $N) -ge $GN ]; then touch $G/runs/$N/DONE; lg "$N DONE"; else lg "$N exited early at gen $(curgen $N)"; fi
    fi
  done < $Q/queue.txt
  # battery cutoff (only when not charging)
  if [ $CH = 0 ] && [ "$LV" -le 15 ] && [ $NR -gt 0 ]; then lg "batt=$LV <=15% stop all"; for f in $Q/run/*.pid; do [ -f "$f" ] && stopj $(basename $f .pid); done; NR=0; fi
  # thermal
  if [ "$T" -ge 470 ] && [ $PAUSED = 0 ]; then for f in $Q/run/*.pid; do [ -f "$f" ] && kill -STOP -$(cat $f) 2>/dev/null; done; PAUSED=1; lg "PAUSE temp=$T"; fi
  if [ $PAUSED = 1 ] && [ "$T" -le 430 ]; then for f in $Q/run/*.pid; do [ -f "$f" ] && kill -CONT -$(cat $f) 2>/dev/null; done; PAUSED=0; lg "resume temp=$T"; fi
  if [ "$T" -ge 440 ] && [ $HOT = 0 ]; then HOT=1; lg "HOT temp=$T -> halve workers"; fi
  if [ "$T" -le 400 ] && [ $HOT = 1 ]; then HOT=0; lg "cool temp=$T -> full workers"; fi
  # fill free slots
  if [ $PAUSED = 0 ] && [ $NR -lt $SLOTS ] && { [ $CH = 1 ] || [ "$LV" -gt 15 ]; }; then
    while IFS=: read N GN A; do [ -z "$N" ] && continue; [ $NR -ge $SLOTS ] && break
      alive $N && continue; [ -f $G/runs/$N/DONE ] && continue; [ -f $G/runs/$N/HOLD ] && continue
      start $N $GN "$A" $W; NR=$((NR+1))
    done < $Q/queue.txt
  fi
  echo "{\"t\":\"$(date +%T)\",\"batt\":$LV,\"temp\":$T,\"charging\":$CH,\"paused\":$PAUSED,\"hot\":$HOT,\"slots\":$SLOTS,\"workers\":$W,\"running\":\"$RUN\"}" > $Q/status.json
  lg "batt=$LV temp=$T ch=$CH paused=$PAUSED hot=$HOT W=$W run:$RUN"
  sleep 30
done
