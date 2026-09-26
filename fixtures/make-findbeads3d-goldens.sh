#!/bin/bash
# Regenerate fixtures/findbeads3d/golden from the native reference findbeads3d.
#
# Inputs in fixtures/findbeads3d: fb_f.mrc (53x47x29, dark beads of
# diameter 6 in Gaussian noise; written as floats) and fb_s.mrc (51x45x27,
# light beads; written as shorts) are seeded synthetic volumes written by the
# native raw2mrc, then converted to bytes by the native `newstack -mode 0
# -scale 0,255` (the originals are in git history before 2026-09-26); tdark.mrc is a 21^3
# float volume holding one dark bead, used as a template; angles.tlt and
# narrow.tlt are tilt-angle files; p.param is a parameter file.  Each case runs
# in its own directory with stdout captured through a pipe, at
# OMP_NUM_THREADS=1; the exit status goes to golden/<case>.rc, stdout to
# golden/<case>.out and output files to golden/<case>/.
#
# `make-findbeads3d-goldens.sh defined` instead writes defined/ from the fixed
# translation ($RSBIN, default target/release/imod) for the cases named in
# defined.list -- those whose output an upstream-bug fix changes (BUGS.md,
# 2026-09-26); golden/ is untouched.
set -e
REF=${REF:-/tmp/imod-reference-build}
HERE=$(cd "$(dirname "$0")/findbeads3d" && pwd)
export AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib OMP_NUM_THREADS=1
PROG=$REF/imodutil/findbeads3d
DEST=golden
if [ "$1" = defined ]; then
  RSBIN=${RSBIN:-$(cd "$HERE/../.." && pwd)/target/release/imod}
  PROG="$RSBIN findbeads3d"
  DEST=defined
  export AUTODOC_DIR=$(cd "$HERE/../.." && pwd)/IMOD/autodoc
fi
rm -rf "$HERE/$DEST"; mkdir -p "$HERE/$DEST"
sed "${FULL:+s/^#full\t//;}/^#/d" "$HERE/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ -z "$name" ] && continue
  [ $DEST = defined ] && ! grep -qx "$name" <(grep -v '^#' "$HERE/defined.list" | cut -f1) && continue
  work=$(mktemp -d)
  cp "$HERE"/*.mrc "$HERE"/*.tlt "$HERE"/*.param "$work"/
  ls "$work" > "$work/.inputs"
  set +e
  (cd "$work" && $PROG $args | cat > "$work/.stdout"; echo "${PIPESTATUS[0]}" > "$work/.rc")
  set -e
  mkdir -p "$HERE/$DEST/$name"
  cp "$work/.stdout" "$HERE/$DEST/$name.out"
  cp "$work/.rc" "$HERE/$DEST/$name.rc"
  for f in "$work"/*; do
    b=$(basename "$f")
    grep -qx "$b" "$work/.inputs" || cp "$f" "$HERE/$DEST/$name/"
  done
  rm -rf "$work"
done
