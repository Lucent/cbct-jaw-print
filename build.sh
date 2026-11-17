#!/bin/sh
# Full build, pieces in parallel: jaws + coupon, then views, stand, checks (parallel), renders.
# Ends with BUILD_DONE, or FAILED <step> at the first failure.  ->  out/build.log
cd "$(dirname "$0")" && PY=${PY:-.venv/bin/python}
exec > out/build.log 2>&1
date
for p in lower upper coupon; do $PY scripts/print_parts.py $p > out/log-$p.txt 2>&1 & eval pid_$p=$!; done
for p in lower upper coupon; do eval wait \$pid_$p || { tail -n 5 out/log-$p.txt; echo "FAILED $p"; exit 1; }; done
$PY scripts/jaw_views.py || { echo "FAILED views"; exit 1; }
$PY scripts/stand.py > out/log-stand.txt 2>&1 || { tail -n 5 out/log-stand.txt; echo "FAILED stand"; exit 1; }
opids=""
for p in cradle arm_right arm_left post base; do $PY scripts/stand_orient.py $p > out/orient-$p.txt 2>&1 & opids="$opids $!"; done
for q in $opids; do wait $q || { echo "FAILED orient"; exit 1; }; done
$PY scripts/stand_print.py > out/log-stand-print.txt 2>&1 || { tail -n 5 out/log-stand-print.txt; echo "FAILED stand print"; exit 1; }
pids=""
for p in lower-jaw upper-jaw detent-0.10 detent-0.20 detent-0.30 detent-0.40; do
  $PY scripts/check_print.py $p > out/check-$p.txt 2>&1 & pids="$pids $!"
done
$PY scripts/render_assembly.py || { echo "FAILED render"; exit 1; }
for q in $pids; do wait $q || { echo "FAILED check"; exit 1; }; done
date
echo BUILD_DONE
