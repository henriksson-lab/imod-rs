#!/bin/bash
# Regenerate fixtures/mrcx/golden from the native reference mrcx (IMOD/mrc/mrcx.c, built by the reference tree).
# Inputs are authored by mrcx/make-inputs.py: MRC files of modes 0,1,2,3,4,16
# (and 12, which mrcx refuses) in both byte orders, new- and old-style
# headers, an odd-sized extended header, trailing bytes, truncated data.
# Each case in mrcx/cases.tsv runs in its own directory holding every input
# (and an empty directory "sub"), stdout captured through a pipe.  "." in the
# arguments column means none.  Exit status goes to golden/<case>.rc, stdout
# to golden/<case>.out, stderr to golden/<case>.err (the test compares only
# whether and how many diagnostic lines appear -- CLAUDE.md, messages), and
# every new or changed file to golden/<case>/.
#
# PROG=<binary> OUT=<dir> runs another binary (e.g. "imod mrcx") into another
# golden directory, for a direct differential.
set -e
REF=${REF:-/tmp/imod-reference-build}
export AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib
HERE=$(cd "$(dirname "$0")/mrcx" && pwd)
INDIR=$HERE
if [ -z "$PROG" ]; then
  PROG=$REF/mrc/mrcx
fi
OUT=${OUT:-$HERE/golden}
rm -rf "$OUT"; mkdir -p "$OUT"
sed "${FULL:+s/^#full\t//;}/^#/d" "$HERE/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ -z "$name" ] && continue
  [ "$args" = "." ] && args=
  work=$(mktemp -d)
  for f in "$INDIR"/*; do case "$f" in *.py|*.tsv) ;; *) [ -f "$f" ] && cp "$f" "$work/";; esac; done
  mkdir "$work/sub"
  set +e
  (cd "$work" && timeout 20 $PROG $args < /dev/null 2> "$work/.stderr" | cat > "$work/.stdout"; echo "${PIPESTATUS[0]}" > "$work/.rc")
  set -e
  mkdir -p "$OUT/$name"
  cp "$work/.stdout" "$OUT/$name.out"
  cp "$work/.stderr" "$OUT/$name.err"
  cp "$work/.rc" "$OUT/$name.rc"
  for f in "$work"/*; do
    b=$(basename "$f")
    [ -d "$f" ] && continue
    if [ -f "$INDIR/$b" ] && cmp -s "$f" "$INDIR/$b"; then continue; fi
    cp "$f" "$OUT/$name/"
  done
  rmdir "$OUT/$name" 2>/dev/null || true
  rm -rf "$work"
done
# Upstream bug fixed in translation (BUGS.md, mrcx): native copies only
# nx*ny*nz bytes of an RGB body to a new file.  golden/new_rgb_{be,le}/
# out_rgb_*.mrc carry the defined output instead -- native's 1048-byte file
# plus the remaining 48 data bytes, i.e. header as native writes it and the
# input's whole 72-byte body (RGB bytes need no swapping).
for e in be le; do
  { cat "$OUT/new_rgb_$e/out_rgb_$e.mrc"; tail -c 48 "$HERE/rgb_$e.mrc"; } > "$OUT/.rgb" &&
    mv "$OUT/.rgb" "$OUT/new_rgb_$e/out_rgb_$e.mrc"
done
