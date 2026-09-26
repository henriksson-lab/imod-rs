#!/bin/bash
# Regenerate fixtures/findsection/golden from the native reference findsection.
#
# Inputs in fixtures/findsection: fz.mrc and m0.mrc are seeded synthetic
# slabs, written by the native raw2mrc in short mode and converted to bytes (half
# the size) by the native `newstack -mode 0 -scale 0,255 <in> <out>` -- the
# short originals are in git history before 2026-09-26; beads.mod is the native
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
sed "${FULL:+s/^#full\t//;}/^#/d" "$HERE/cases.tsv" | while IFS=$'\t' read -r name args; do
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
# Upstream bugs fixed in translation (BUGS.md, `findsection`): the table's
# binning column prints mBinning[scl][1] twice, and lowSDerrStrings is indexed
# with the 1-based error code.  The goldens carry the defined behaviour.
sed -i "s/^\\(  2    2,2,\\)2 /\\11 /" "$HERE/golden/binning.out"
sed -i 's/^   No minimum in median SD value found on one side of peak$/   No peak in median SD value found except at top or bottom/' "$HERE/golden/lowest.out"
# Upstream bug fixed in translation (BUGS.md, `unit_header.c:256-258`): native
# sets xlen from mxyz[1] and leaves ylen alone; the defined cell lengths are
# (mx, my), so bytes 40-47 of the -volume outputs are rewritten from 28-35.
python3 - "$HERE"/golden/volume/*.colmed <<'PY'
import struct, sys
for path in sys.argv[1:]:
    with open(path, 'r+b') as f:
        head = f.read(36)
        f.seek(40)
        f.write(struct.pack('<2f', *struct.unpack('<2i', head[28:36])))
PY
