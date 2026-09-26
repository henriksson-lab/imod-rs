#!/bin/bash
# Regenerates the tomopitch model inputs from points/*.txt with the native
# point2model (open contours; -times for tm*, a pixel spacing and origin for
# scaled).  bad.mod is a text file that is not a model.
set -e
cd "$(dirname "$0")"
R=${IMOD_REF:-/tmp/imod-reference-build}
export LD_LIBRARY_PATH=$R/buildlib AUTODOC_DIR=$R/autodoc
for f in points/*.txt; do
  n=$(basename "$f" .txt)
  case $n in whole*|crossed*|diag) v=1000,500,500;; scaled) v=500,250,20;; *) v=1000,500,20;; esac
  extra=""
  case $n in tm*) extra=-times;; scaled) extra="-pixel 2,2,2 -origin 10,20,5";; esac
  $R/imodutil/point2model -volume $v -open $extra -input "$f" -output $n.mod > /dev/null
done
echo garbage > bad.mod
