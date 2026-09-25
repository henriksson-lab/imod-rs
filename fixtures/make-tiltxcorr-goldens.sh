#!/bin/bash
# Regenerate fixtures/tiltxcorr/golden from the native reference tiltxcorr.
#
# Inputs in fixtures/tiltxcorr: tx.st is a seeded synthetic tilt series (96 x 88,
# 13 views at -60..60 in 10-degree steps, tilt axis at -6 degrees, per-view random
# shifts and noise, short mode) written by the native raw2mrc from the generator in
# /big/henriksson/realbench/wave2-2C/make_series.py; tx.tlt its angles; tx.prexf the
# native `tiltxcorr -input tx.st -output tx.prexf -tiltfile tx.tlt -rotation -6`
# output; bound.mod/boundw.mod/seed.mod small hand-written ASCII models (a closed
# contour on the zero-tilt view, closed contours on views 3/7/11, scattered seed
# points); tx.pl a trivial piece list.  Each case runs in its own directory with
# stdout captured through a pipe, at OMP_NUM_THREADS=1; the exit status goes to
# golden/<case>.rc, stdout to golden/<case>.out and output files to golden/<case>/.
set -e
REF=${REF:-/tmp/imod-reference-build}
HERE=$(cd "$(dirname "$0")/tiltxcorr" && pwd)
export AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib OMP_NUM_THREADS=1
rm -rf "$HERE/golden"; mkdir -p "$HERE/golden"
grep -v '^#' "$HERE/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ -z "$name" ] && continue
  work=$(mktemp -d)
  cp "$HERE"/tx.* "$HERE"/*.mod "$work"/
  ls "$work" > "$work/.inputs"
  set +e
  (cd "$work" && $REF/imodutil/tiltxcorr $args | cat > "$work/.stdout"; echo "${PIPESTATUS[0]}" > "$work/.rc")
  set -e
  mkdir -p "$HERE/golden/$name"
  cp "$work/.stdout" "$HERE/golden/$name.out"
  cp "$work/.rc" "$HERE/golden/$name.rc"
  for f in "$work"/*; do
    b=$(basename "$f")
    grep -qx "$b" "$work/.inputs" || cp "$f" "$HERE/golden/$name/"
  done
  rm -rf "$work"
done
