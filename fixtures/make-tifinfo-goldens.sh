#!/bin/bash
# Regenerate fixtures/tifinfo/golden from a native tifinfo compiled from the vendored
# IMOD/mrc/tifinfo.c (upstream's Makefile does not build it).  Inputs are
# authored by tifinfo/make-inputs.py.
# Each case in tifinfo/cases.tsv runs in its own directory holding every input
# (and an empty directory "sub"), stdout captured through a pipe.  "." in the
# arguments column means none.  Exit status goes to golden/<case>.rc, stdout
# to golden/<case>.out, stderr to golden/<case>.err (the test compares only
# whether and how many diagnostic lines appear -- CLAUDE.md, messages), and
# every new or changed file to golden/<case>/.
#
# BUGS.md, fixed in translation: native tifinfo.c assumes a big-endian host.
# Every case whose golden differs between native and the translation (all
# cases that walk IFD entries, plus badver/noifd, whose version word is now
# read in the file's order) was written by the translation with
# PROG="imod tifinfo", after checking every real_*.tif against libtiff's
# `tiffdump` (tag, type, count, inline SHORT/LONG values and -v data):
#   badver v_badver iibe_* v_iibe_* mmle_* v_mmle_* noifd v_noifd several
#   quiet_several missing_middle real_* v_real_*
# The remaining cases are native goldens.
#
# PROG=<binary> OUT=<dir> runs another binary (e.g. "imod tifinfo") into another
# golden directory, for a direct differential.
set -e
REF=${REF:-/tmp/imod-reference-build}
export AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib
HERE=$(cd "$(dirname "$0")/tifinfo" && pwd)
INDIR=$HERE
if [ -z "$PROG" ]; then
  BIN=$(mktemp -d)
  gcc -m64 -O3 -w -D_FILE_OFFSET_BITS=64 -I$REF/include -o $BIN/tifinfo $REF/mrc/tifinfo.c \
      -L$REF/buildlib -liimod -lcfshr -limxml -ltiff -ljpeg -fopenmp -lc -lm
  PROG=$BIN/tifinfo
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
