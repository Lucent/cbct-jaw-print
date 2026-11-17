#!/bin/sh
# Rebuild the print from the scan, in this folder (the scripts read and write here and in out/):
#   ./reproduce.sh SCAN_DIR | jaw_0p1403mm.nrrd[.gz]  [SEG]  [DIRS]
# SEG: a saved TotalSegmentator labelling to use instead of running it; DIRS: a saved out/dirs.json (the removal
# directions; without it they are worked out afresh). Needs PY (default .venv/bin/python) with env/requirements.txt,
# TotalSegmentator beside it, and its teeth weights (model/). Ends with REPRODUCE_DONE.
set -e
cd "$(dirname "$0")"; PY=${PY:-.venv/bin/python}; BIN=$(dirname "$PY")
case "$1" in
  *.nrrd.gz) cp "$1" jaw_0p1403mm.nrrd.gz ;;
  *.nrrd) gzip -nc "$1" > jaw_0p1403mm.nrrd.gz ;;
  *) $PY ingest_jpeg.py "$1"; gzip -nf jaw_0p1403mm.nrrd ;;
esac
$PY to_nifti.py
if [ -n "$2" ]; then cp "$2" seg_teeth.nii.gz; else "$BIN/TotalSegmentator" -i jaw.nii.gz -o seg_teeth.nii.gz --ml -ta teeth > seg.log 2>&1; fi
$PY scripts/refine2.py > refine2.log 2>&1
mkdir -p out
[ -n "$3" ] && cp "$3" out/dirs.json
$PY scripts/landmarks.py
sh build.sh; tail -n 1 out/build.log | grep -q BUILD_DONE
echo REPRODUCE_DONE
