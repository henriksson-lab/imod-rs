#!/bin/bash
# Regenerate fixtures/<prog>/golden from the native reference program, for
# prog = ctfphaseflip (this script's name) or mtffilter (make-mtffilter-goldens.sh,
# a link to this file).
#
# The inputs in fixtures/<prog> are written by make-ctfphaseflip-mtffilter-inputs.py
# (seeded synthetic stacks through the native raw2mrc, the FFT through the native
# `clip fft -3d`).  Each case of fixtures/<prog>/cases.tsv runs in its own
# directory holding a copy of the inputs; a case may have several steps separated
# by ' ;; ' (a parallel-writing setup run and its chunks), run in order in that
# directory.  Stdout is captured through a pipe, with each step followed by a
# line `rc=<status>`; it goes to golden/<case>.out, the last step's exit status to
# golden/<case>.rc, and every file the case created or rewrote to golden/<case>/.
set -e
REF=${REF:-/tmp/imod-reference-build}
PROG=$(basename "$0" .sh); PROG=${PROG#make-}; PROG=${PROG%-goldens}
HERE=$(cd "$(dirname "$0")/$PROG" && pwd)
export AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib OMP_NUM_THREADS=1
unset IMOD_USE_GPU IMOD_OUTPUT_FORMAT
rm -rf "$HERE/golden"; mkdir -p "$HERE/golden"
sed "${FULL:+s/^#full\t//;}/^#/d" "$HERE/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ -z "$name" ] && continue
  work=$(mktemp -d)
  find "$HERE" -maxdepth 1 -type f ! -name cases.tsv ! -name golden.manifest -exec cp {} "$work"/ \;
  # mtffilter runs on ctfphaseflip's f48.mrc (not stored twice).
  [ $PROG = mtffilter ] && cp "$HERE/../ctfphaseflip/f48.mrc" "$work"/
  ls "$work" > "$work/.inputs"
  set +e
  echo "$args" | sed 's/ ;; /\n/g' | while read -r step; do
    (cd "$work" && $REF/mrc/$PROG $step | cat; echo "rc=${PIPESTATUS[0]}")
  done > "$work/.stdout"
  set -e
  mkdir -p "$HERE/golden/$name"
  cp "$work/.stdout" "$HERE/golden/$name.out"
  tail -1 "$work/.stdout" | sed 's/^rc=//' > "$HERE/golden/$name.rc"
  for f in "$work"/*; do
    b=$(basename "$f")
    # New files, and inputs the case rewrote (mtffilter without an output file)
    src=$HERE/$b
    [ $PROG = mtffilter ] && [ $b = f48.mrc ] && src=$HERE/../ctfphaseflip/f48.mrc
    if ! grep -qx "$b" "$work/.inputs" || ! cmp -s "$f" "$src"; then
      cp "$f" "$HERE/golden/$name/"
    fi
  done
  rm -rf "$work"
done
