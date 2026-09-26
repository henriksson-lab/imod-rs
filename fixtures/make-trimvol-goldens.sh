#!/bin/bash
# Regenerate fixtures/trimvol/golden from the native Python trimvol
# (IMOD/pysrc/trimvol) driving the native reference densmatch, findcontrast,
# newstack, clip and header.  Inputs are fixtures/densmatch/*.mrc.  Each case
# runs in its own directory with stdout captured through a pipe; exit status
# goes to golden/<case>.rc, stdout to golden/<case>.out, and new files to
# golden/<case>/.
set -e
REF=${REF:-/tmp/imod-reference-build}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HERE=$ROOT/fixtures/trimvol
BIN=$(mktemp -d)
for p in flib/image/densmatch flib/image/findcontrast flib/image/newstack clip/clip flib/image/header; do
  ln -s "$REF/$p" "$BIN/$(basename $p)"
done
export AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib OMP_NUM_THREADS=1
export IMOD_DIR=$REF PYTHONPATH=$ROOT/IMOD/pysrc PATH=$BIN:$PATH
rm -rf "$HERE/golden"; mkdir -p "$HERE/golden"
sed "${FULL:+s/^#full\t//;}/^#/d" "$HERE/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ -z "$name" ] && continue
  work=$(mktemp -d)
  cp "$ROOT"/fixtures/densmatch/*.mrc "$work"/
  ls "$work" > "$work/.inputs"
  set +e
  (cd "$work" && python3 "$ROOT/IMOD/pysrc/trimvol" $args | cat > "$work/.stdout"; echo "${PIPESTATUS[0]}" > "$work/.rc")
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
rm -rf "$BIN"
