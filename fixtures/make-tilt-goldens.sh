#!/bin/bash
# Regenerate fixtures/tilt/golden from the native reference tilt.
#
# Inputs in fixtures/tilt: t.ali is a seeded synthetic aligned tilt series (short
# mode, 21 views -60..60 step 6) written by the native raw2mrc; t.tlt, t.xtilt,
# t.zfac and t.local are the matching seeded tilt, X-tilt, Z-factor and local
# alignment files; t.rec is `tilt -input t.ali -output t.rec -TILTFILE t.tlt
# -THICKNESS 16 -MODE 1 -SCALE 0,20` from the native tilt; pm.mod is the native
# `point2model -sizes -scat` of six seeded points.  Each case runs in its own
# directory with stdout captured through a pipe; the exit status goes to
# golden/<case>.rc, stdout to golden/<case>.out and output files to golden/<case>/.
# An argument column STDIN:<file> runs `tilt -StandardInput < <file>`.
set -e
REF=${REF:-/tmp/imod-reference-build}
HERE=$(cd "$(dirname "$0")/tilt" && pwd)
export AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib OMP_NUM_THREADS=1 IMOD_NO_IMAGE_BACKUP=1
rm -rf "$HERE/golden"; mkdir -p "$HERE/golden"
grep -v '^#' "$HERE/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ -z "$name" ] && continue
  work=$(mktemp -d)
  cp "$HERE"/t.* "$HERE"/pm.mod "$HERE"/std.com "$work"/
  ls "$work" > "$work/.inputs"
  set +e
  if [[ $args == STDIN:* ]]; then
    (cd "$work" && $REF/flib/tilt/tilt -StandardInput < "${args#STDIN:}" | cat > "$work/.stdout"; echo "${PIPESTATUS[0]}" > "$work/.rc")
  else
    (cd "$work" && $REF/flib/tilt/tilt $args | cat > "$work/.stdout"; echo "${PIPESTATUS[0]}" > "$work/.rc")
  fi
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
