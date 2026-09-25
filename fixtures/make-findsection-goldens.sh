#!/bin/bash
# Regenerate fixtures/findsection/golden from the native reference findsection.
#
# Inputs in fixtures/findsection: fz.mrc/fy.mrc/m0.mrc/m1.mrc are seeded synthetic
# slabs (short mode) written by the native raw2mrc; beads.mod is the native
# `findsection -tomo fz.mrc -size 8,8,2 -block 16 -point bb` column-boundary model
# passed through the native `imodtrans -i fz.mrc` (which sets IMODF_OTRANS_ORIGIN);
# noref.mod is fixtures/imodtrans/multi.mod (no IrefImage).  Each case runs in its
# own directory with stdout captured through a pipe; the exit status goes to
# golden/<case>.rc, stdout to golden/<case>.out and output files to golden/<case>/.
set -e
REF=${REF:-/tmp/imod-reference-build}
HERE=$(cd "$(dirname "$0")/findsection" && pwd)
export AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib OMP_NUM_THREADS=1
rm -rf "$HERE/golden"; mkdir -p "$HERE/golden"
grep -v '^#' "$HERE/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ -z "$name" ] && continue
  work=$(mktemp -d)
  cp "$HERE"/*.mrc "$HERE"/*.mod "$work"/
  ls "$work" > "$work/.inputs"
  set +e
  (cd "$work" && $REF/imodutil/findsection $args | cat > "$work/.stdout"; echo "${PIPESTATUS[0]}" > "$work/.rc")
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
