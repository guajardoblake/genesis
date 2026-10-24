G=/data/local/tmp/gx; Q=$G/q
cp /sdcard/gxq/guard3.sh $G/ 2>/dev/null; chmod 755 $G/guard3.sh
cd $G
setsid sh $G/guard3.sh < /dev/null > $Q/g3.out 2>&1 & sleep 3
echo launched
