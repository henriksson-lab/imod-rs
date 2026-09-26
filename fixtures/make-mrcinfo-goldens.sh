#!/bin/bash
# Regenerate fixtures/mrcinfo/golden from a native mrcinfo compiled from the vendored
# IMOD/mrc/mrcinfo.c (upstream's Makefile does not build it) against the
# reference libiimod.  Inputs are fixtures/mrcx/*.
# Each case in mrcinfo/cases.tsv runs in its own directory holding every input
# (and an empty directory "sub"), stdout captured through a pipe.  "." in the
# arguments column means none.  Exit status goes to golden/<case>.rc, stdout
# to golden/<case>.out, stderr to golden/<case>.err (the test compares only
# whether and how many diagnostic lines appear -- CLAUDE.md, messages), and
# every new or changed file to golden/<case>/.
#
# PROG=<binary> OUT=<dir> runs another binary (e.g. "imod mrcinfo") into another
# golden directory, for a direct differential.
set -e
REF=${REF:-/tmp/imod-reference-build}
export AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib
HERE=$(cd "$(dirname "$0")/mrcinfo" && pwd)
INDIR=$(cd "$(dirname "$0")/mrcx" && pwd)
if [ -z "$PROG" ]; then
  BIN=$(mktemp -d)
  gcc -m64 -O3 -w -D_FILE_OFFSET_BITS=64 -I$REF/include -o $BIN/mrcinfo $REF/mrc/mrcinfo.c \
      -L$REF/buildlib -liimod -lcfshr -limxml -ltiff -ljpeg -fopenmp -lc -lm
  PROG=$BIN/mrcinfo
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
