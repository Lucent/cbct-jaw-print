#!/bin/sh
# Jaws only (both in parallel, GPU) and their close-up views: for iterating on the anatomy.  ->  out/jaws.log
# Ends with JAWS_DONE, or FAILED <step> at the first failure (nothing after it runs on stale meshes).
cd "$(dirname "$0")" && PY=${PY:-.venv/bin/python}
exec > out/jaws.log 2>&1
date
$PY scripts/print_parts.py lower > out/log-lower.txt 2>&1 & L=$!
$PY scripts/print_parts.py upper > out/log-upper.txt 2>&1 & U=$!
wait $L || { tail -n 5 out/log-lower.txt; echo "FAILED lower"; exit 1; }
wait $U || { tail -n 5 out/log-upper.txt; echo "FAILED upper"; exit 1; }
tail -n 2 out/log-lower.txt out/log-upper.txt
$PY scripts/jaw_views.py || { echo "FAILED views"; exit 1; }
date
echo JAWS_DONE
