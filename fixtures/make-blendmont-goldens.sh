#!/bin/bash
# Regenerate fixtures/blendmont/golden from the native reference blendmont.
#
# Inputs in fixtures/blendmont come from make-blendmont-inputs.py.  First the
# native blendmont makes the "old" edge files other cases reuse
# (`-rootname old -sloppy -intensity 2`: old.xef/.yef/.xaed/.yaed/.ecd).  Each
# case of cases.tsv then runs in its own directory with stdout captured
# through a pipe, at OMP_NUM_THREADS=1, with the reference `clip` first on
# PATH (for `-sum`'s `clip plane`); the exit status goes to golden/<case>.rc,
# stdout to golden/<case>.out and the files it wrote to golden/<case>/.
set -e
REF=${REF:-/tmp/imod-reference-build}
HERE=$(cd "$(dirname "$0")/blendmont" && pwd)
export AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib OMP_NUM_THREADS=1
export PATH=$REF/clip:$PATH
rm -f "$HERE"/old.*
work=$(mktemp -d)
cp "$HERE"/bm.st "$HERE"/bm.pl "$work"/
(cd "$work" && $REF/flib/blend/blendmont -imin bm.st -plin bm.pl -imout seed.st -rootname old -sloppy -intensity 2 > /dev/null)
cp "$work"/old.* "$HERE"/
rm -rf "$work"
rm -rf "$HERE/golden"; mkdir -p "$HERE/golden"
grep -v '^#' "$HERE/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ -z "$name" ] && continue
  work=$(mktemp -d)
  cp "$HERE"/bm.st "$HERE"/*.pl "$HERE"/*.mod "$HERE"/g.txt "$HERE"/x.xf "$HERE"/old.* "$work"/
  ls "$work" > "$work/.inputs"
  set +e
  (cd "$work" && $REF/flib/blend/blendmont $args | cat > "$work/.stdout"; echo "${PIPESTATUS[0]}" > "$work/.rc")
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
