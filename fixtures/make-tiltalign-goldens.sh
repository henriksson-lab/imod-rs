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
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
F=$(cd "$(dirname "$0")/tiltalign" && pwd)
S=$(mktemp -d)
rm -f "$F"/golden/*
sed "${FULL:+s/^#full\t//;}/^#/d" "$F/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ "$args" = "-" ] && args=""
  d="$S/$name"; mkdir "$d"
  find "$F" -maxdepth 1 -type f ! -name cases.tsv ! -name gen6.py ! -name golden.manifest -exec cp {} "$d"/ \;
  ls "$d" > "$S/inputs"
  set +e
  (cd "$d" && OMP_NUM_THREADS=1 AUTODOC_DIR=$R/autodoc LD_LIBRARY_PATH=$R/buildlib \
      timeout 300 $R/flib/tiltalign/tiltalign $args < /dev/null 2>/dev/null \
      | cat > "$S/stdout"; exit ${PIPESTATUS[0]})
  echo $? > "$F/golden/$name.rc"; set -e
  cp "$S/stdout" "$F/golden/$name.stdout"
  for o in $(ls "$d"); do
    grep -qx "$o" "$S/inputs" || cp "$d/$o" "$F/golden/$name.$o"
  done
done
rm -rf "$S"
