#!/bin/bash
# Regenerate fixtures/beadtrack/golden from the native reference beadtrack.
#
# Inputs in fixtures/beadtrack: t1.mrc is a seeded synthetic tilt series (127 x 121,
# 21 views at -60..60 in 6-degree steps, 7 dark Gaussian beads projected from random
# 3D positions with a 4-degree tilt axis, per-view shifts, small mag changes, noise,
# and prealignment blanking at the edges) written as shorts by the native raw2mrc
# from /big/henriksson/realbench/wave6-beadtrack/gen.py (`gen.py t1 127 121 -60 60 6 7
# 21 short`), then converted to bytes by the native `newstack -mode 0 -scale 0,255`
# (2026-09-26; the short original is in git history); t1.rawtlt its angles; t1.prexf
# the blanking shifts; t1.seed/t1.seed2 the
# native point2model seed (one object; beads split over two objects) with one point
# per bead on the zero-tilt view; t1.dupseed two points on one view (error path).
# cases/skipedge.in (SkipViews 1-10, two local areas, two rounds) reaches the
# mResMean overrun (BUGS.md, beadtrack, 2026-09-27); native matches the defined
# behaviour on it, so it is not in defined.list.
# Each case is the PIP input in cases/<case>.in, run as `beadtrack -StandardInput`
# in its own directory with stdout captured through a pipe, at OMP_NUM_THREADS=1;
# the exit status goes to golden/<case>.rc, stdout to golden/<case>.out and output
# files to golden/<case>/.
#
# `make-beadtrack-goldens.sh defined` then overwrites golden/<case> with the fixed
# translation's output ($RSBIN, default target/release/imod) for the cases named in
# defined.list -- those an upstream-bug fix changes (BUGS.md, beadtrack,
# 2026-09-26).  A build with only the Sobel peak-scaling fix reverted reproduces
# native on every Sobel case, so that fix is the whole difference there.
set -e
REF=${REF:-/tmp/imod-reference-build}
HERE=$(cd "$(dirname "$0")/beadtrack" && pwd)
export AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib OMP_NUM_THREADS=1
PROG=$REF/flib/beadtrack/beadtrack
if [ "$1" = defined ]; then
  RSBIN=${RSBIN:-$(cd "$HERE/../.." && pwd)/target/release/imod}
  PROG="$RSBIN beadtrack"
  export AUTODOC_DIR=$(cd "$HERE/../.." && pwd)/IMOD/autodoc
else
  rm -rf "$HERE/golden"; mkdir -p "$HERE/golden"
fi
for input in "$HERE"/cases/*.in; do
  name=$(basename "$input" .in)
  if [ "$1" = defined ]; then
    grep -qx "$name" <(grep -v '^#' "$HERE/defined.list" | cut -f1) || continue
    rm -rf "$HERE/golden/$name" "$HERE/golden/$name.out" "$HERE/golden/$name.rc"
  fi
  work=$(mktemp -d)
  cp "$HERE"/t1.* "$work"/
  ls "$work" > "$work/.inputs"
  set +e
  (cd "$work" && $PROG -StandardInput < "$input" | cat > "$work/.stdout"; echo "${PIPESTATUS[0]}" > "$work/.rc")
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
