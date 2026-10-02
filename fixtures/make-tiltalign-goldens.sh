#!/bin/bash
# Regenerate fixtures/tiltalign/golden/ from the native reference tiltalign.
# Inputs: fiducial models g1..g4.fid made from gen6.py's point2model text
# (python3 gen6.py, then native `point2model -zcoord -open gN.txt gN.fid`, with
# `-volume 2048,2048,61` for g2) and the matching nominal-angle .rawtlt files;
# few.xyz is 9 fixed-XYZ lines for a 30-point model.  For each row of cases.tsv
# the native program runs in a fresh copy of the inputs with stdout captured
# through a pipe; golden/<name>.rc holds the exit status, .stdout the standard
# output, and golden/<name>.<file> every file the run created.
# CrossValidate is always given: when it is absent the source reads an
# uninitialised int and cross-validates in about half the runs (BUGS.md).
#
# pt6.fid is every sixth contour (31 of 184) of the chopped TS_01 patch-tracking
# model fixtures/restrictalign/pt.fid, pt6.rawtlt the TS_01 angles.  Its case
# pt6_trackgroup_cv (robust fitting by whole tracks with cross-validation) is
# in FIXED_CASES: native reads past the collected track residuals there
# (BUGS.md, "tiltalign: the track-group median counts tracks it did not
# collect"), so its golden comes from $TA_FIXED, the reference tiltalign.cpp
# rebuilt with that fix alone (`ninViewSum += 1` moved inside `if (ninTrack)`,
# tiltalign.cpp:2513 -> 2525; built in a copy of the reference flib/tiltalign;
# the unpatched rebuild is byte-identical to the reference binary).  Stock
# native differs from it in the robust leave-out error and benefit lines.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
TA_FIXED=${TA_FIXED:-/big/henriksson/realbench/brtmissing/tapatch/tiltalign}
FIXED_CASES=" pt6_trackgroup_cv "
F=$(cd "$(dirname "$0")/tiltalign" && pwd)
S=$(mktemp -d)
mkdir -p "$F/golden"; rm -f "$F"/golden/*
sed "${FULL:+s/^#full\t//;}/^#/d" "$F/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ "$args" = "-" ] && args=""
  d="$S/$name"; mkdir "$d"
  find "$F" -maxdepth 1 -type f ! -name cases.tsv ! -name gen6.py ! -name golden.manifest -exec cp {} "$d"/ \;
  ls "$d" > "$S/inputs"
  prog=$R/flib/tiltalign/tiltalign
  if [[ $FIXED_CASES == *" $name "* ]]; then
    [ -x "$TA_FIXED" ] || { echo "$name needs the fixed tiltalign TA_FIXED=$TA_FIXED" >&2; exit 1; }
    prog=$TA_FIXED
  fi
  set +e
  (cd "$d" && OMP_NUM_THREADS=1 AUTODOC_DIR=$R/autodoc LD_LIBRARY_PATH=$R/buildlib \
      timeout 300 $prog $args < /dev/null 2>/dev/null \
      | cat > "$S/stdout"; exit ${PIPESTATUS[0]})
  echo $? > "$F/golden/$name.rc"; set -e
  cp "$S/stdout" "$F/golden/$name.stdout"
  for o in $(ls "$d"); do
    grep -qx "$o" "$S/inputs" || cp "$d/$o" "$F/golden/$name.$o"
  done
done
rm -rf "$S"
