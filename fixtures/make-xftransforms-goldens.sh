#!/bin/bash
# Regenerate fixtures/xftransforms/golden/ from the native reference xftoxg and
# xfproduct.  Inputs: the generated *.xf here (linear lists written by a seeded
# Python script with rotations/stretches/shifts; grid and control-point warping
# files in warpfiles.c's version-3 layout), plus BBa.xf, midzone2a.xf and
# mediumscansb2_midas.xf from the vendored tree.  For each case of cases.tsv the
# native program runs in a fresh directory with stdout captured through a pipe;
# golden/<prog>-<name>.rc holds the exit status, .stdout the standard output and
# .out the output transform file when native leaves one (o.xg, BBa.xg or p.xf).
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
F=$(cd "$(dirname "$0")/xftransforms" && pwd)
V="$F/../../IMOD/Etomo/uitestData"
S=$(mktemp -d)
rm -f "$F"/golden/*
grep -v '^#' "$F/cases.tsv" | while IFS=$'\t' read -r prog name args stdin; do
  [ "$args" = "-" ] && args=""
  d="$S/$prog-$name"; mkdir "$d"
  cp "$F"/*.xf "$V/BB/BBa.xf" "$V/midzone2/midzone2a.xf" "$V/mediumscansb2/mediumscansb2_midas.xf" "$d"/
  [ "$stdin" = "-" ] && stdin=""
  set +e
  (cd "$d" && printf -- "$stdin" | AUTODOC_DIR=$R/autodoc LD_LIBRARY_PATH=$R/buildlib \
      timeout 60 $R/flib/image/$prog $args 2>/dev/null | cat > "$S/stdout"; exit ${PIPESTATUS[1]})
  echo $? > "$F/golden/$prog-$name.rc"; set -e
  cp "$S/stdout" "$F/golden/$prog-$name.stdout"
  for o in o.xg BBa.xg p.xf; do
    if [ -f "$d/$o" ]; then cp "$d/$o" "$F/golden/$prog-$name.out"; fi
  done
done
rm -rf "$S"
