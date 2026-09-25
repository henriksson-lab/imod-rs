#!/bin/bash
# Regenerate fixtures/{densmatch,findcontrast}/golden from the native reference
# programs.  The inputs (f.mrc/f2.mrc float, s.mrc short, b.mrc byte; 32x24x6,
# seeded synthetic content) were written by the native raw2mrc.  Each case in
# <prog>/cases.tsv runs in its own directory with stdout captured through a
# pipe; "." in the arguments column means none, and a third column, when present, is fed on stdin (`\n` for
# newlines).  Exit status goes to golden/<case>.rc, stdout to
# golden/<case>.out, and every new or changed file to golden/<case>/.
set -e
REF=${REF:-/tmp/imod-reference-build}
export AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib OMP_NUM_THREADS=1
for prog in densmatch findcontrast; do
  HERE=$(cd "$(dirname "$0")/$prog" && pwd)
  rm -rf "$HERE/golden"; mkdir -p "$HERE/golden"
  grep -v '^#' "$HERE/cases.tsv" | while IFS=$'\t' read -r name args stdin; do
    [ -z "$name" ] && continue
    [ "$args" = "." ] && args=
    work=$(mktemp -d)
    cp "$HERE"/*.mrc "$work"/
    set +e
    (cd "$work" && printf "$stdin" | $REF/flib/image/$prog $args | cat > "$work/.stdout"; echo "${PIPESTATUS[1]}" > "$work/.rc")
    set -e
    mkdir -p "$HERE/golden/$name"
    cp "$work/.stdout" "$HERE/golden/$name.out"
    cp "$work/.rc" "$HERE/golden/$name.rc"
    for f in "$work"/*; do
      b=$(basename "$f")
      if [ -f "$HERE/$b" ] && cmp -s "$f" "$HERE/$b"; then continue; fi
      cp "$f" "$HERE/golden/$name/"
    done
    rm -rf "$work"
  done
done
