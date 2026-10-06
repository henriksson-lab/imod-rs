#!/bin/bash
# Regenerate fixtures/findsection/golden from the native reference findsection.
#
# Inputs in fixtures/findsection: m0.mrc is a seeded synthetic slab (fz.mrc,
# the 139 KB Z slab, its bead models and its 15 cases were deleted on
# 2026-10-06), written by the native raw2mrc in short mode and converted to
# bytes (half the size) by the native `newstack -mode 0 -scale 0,255 <in> <out>` -- the
# short originals are in git history before 2026-09-26.  Each case runs in its
# own directory with stdout captured through a pipe; the exit status goes to
# golden/<case>.rc, stdout to golden/<case>.out and output files to golden/<case>/.
set -e
REF=${REF:-/tmp/imod-reference-build}
HERE=$(cd "$(dirname "$0")/findsection" && pwd)
export AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib OMP_NUM_THREADS=1
rm -rf "$HERE/golden"; mkdir -p "$HERE/golden"
sed "${FULL:+s/^#full\t//;}/^#/d" "$HERE/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ -z "$name" ] && continue
  work=$(mktemp -d)
  cp "$HERE"/*.mrc "$work"/
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
# (The sed/python fixups for the defined behaviour of three upstream bugs --
# the binning column, lowSDerrStrings and the -volume cell lengths -- went with
# the fz.mrc cases `binning`, `lowest` and `volume` on 2026-10-06.)
