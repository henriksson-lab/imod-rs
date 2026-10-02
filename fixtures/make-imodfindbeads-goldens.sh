#!/bin/bash
# Regenerate fixtures/imodfindbeads/ (inputs and golden/) from the native
# reference imodfindbeads.
#
# Inputs are authored here, seeded: fb.mrc (160x160x3 bytes) holds ~55 dark
# gold-like beads of diameter 8 per section at random, well-spaced positions
# on a noisy background with a gentle ramp, written by the native raw2mrc;
# area.mod (two closed boundary contours on sections 0 and 1) and ref.mod
# (boundary object 1 plus the seeded bead centres as object 2) come from the
# native point2model; fb.prexg is a prealignment transform file; fb.tlt the
# tilt angles; add.mod is native imodfindbeads' own output for the strongest
# beads on sections 0 and 1, used as the model to add to.  For each row of cases.tsv the program
# runs in a fresh directory holding the inputs, stdout captured through a
# pipe; golden/<case>.rc is the exit status, golden/<case>.out the standard
# output and golden/<case>/ every file it left behind.
#
# `make-imodfindbeads-goldens.sh defined` instead writes defined/ from the
# fixed translation ($RSBIN, default target/release/imod) for the cases named
# in defined.list -- those whose output an upstream-bug fix changes (BUGS.md,
# imodfindbeads); golden/ is untouched.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
F=$(cd "$(dirname "$0")/imodfindbeads" && pwd)
export AUTODOC_DIR=$R/autodoc LD_LIBRARY_PATH=$R/buildlib
PROG=$R/imodutil/imodfindbeads
DEST=golden
if [ "$1" = defined ]; then
  RSBIN=${RSBIN:-$(cd "$F/../.." && pwd)/target/release/imod}
  PROG="$RSBIN imodfindbeads"
  DEST=defined
else
  S=$(mktemp -d)
  cd "$S"
  python3 - <<'PY'
import numpy as np
r = np.random.default_rng(11)
nx = ny = 160
nz = 3
vol = np.zeros((nz, ny, nx), np.float32)
yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
pts = []
for z in range(nz):
    img = 120. + 0.05 * xx - 0.03 * yy + r.normal(0, 9, (ny, nx))
    placed = []
    tries = 0
    while len(placed) < 55 and tries < 5000:
        tries += 1
        x, y = r.uniform(6, nx - 6), r.uniform(6, ny - 6)
        if all((x - a) ** 2 + (y - b) ** 2 > 12 ** 2 for a, b in placed):
            placed.append((x, y))
    for (x, y) in placed:
        d = np.sqrt((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2)
        img -= r.uniform(40, 70) / (1. + np.exp(np.minimum((d - 4.) / 0.7, 50.)))
        pts.append((z, x, y))
    vol[z] = img
np.clip(vol, 0, 255).astype(np.uint8).tofile('fb.raw')
with open('ref.txt', 'w') as f:
    f.write('1 1 10 10 0\n1 1 150 12 0\n1 1 140 150 0\n1 1 12 140 0\n')
    f.write('1 2 30 30 1\n1 2 130 30 1\n1 2 130 130 1\n1 2 30 130 1\n')
    for k, (z, x, y) in enumerate(pts):
        f.write('2 %d %.2f %.2f %d\n' % (k + 1, x, y, z))
with open('area.txt', 'w') as f:
    f.write('1 1 20 20 0\n1 1 140 25 0\n1 1 120 140 0\n1 1 25 120 0\n')
    f.write('1 2 40 40 1\n1 2 120 40 1\n1 2 120 120 1\n1 2 40 120 1\n')
    f.write('1 3 60 60 2\n1 3 100 60 2\n1 3 100 100 2\n1 3 60 100 2\n')
with open('fb.prexg', 'w') as f:
    for z, (dx, dy) in enumerate([(6.5, -4.0), (0.0, 0.0), (-9.25, 7.5)]):
        f.write('%12.7f%12.7f%12.7f%12.7f%12.3f%12.3f\n' % (1, 0, 0, 1, dx, dy))
with open('fb.tlt', 'w') as f:
    f.write('-3.0\n0.0\n3.0\n')
PY
  $R/mrc/raw2mrc -x 160 -y 160 -z 3 -t byte fb.raw fb.mrc > /dev/null
  $R/imodutil/point2model -zc area.txt area.mod > /dev/null
  $R/imodutil/point2model -zc ref.txt ref.mod > /dev/null
  $PROG -input fb.mrc -output add.mod -size 8 -sections 0,1 -store 0.6 > /dev/null
  rm -f "$F"/*.mrc "$F"/*.mod "$F"/*.prexg "$F"/*.tlt
  cp fb.mrc area.mod ref.mod add.mod fb.prexg fb.tlt "$F"/
  cd "$F"; rm -rf "$S"
fi
rm -rf "$F/$DEST"; mkdir -p "$F/$DEST"
sed "${FULL:+s/^#full\t//;}/^#/d" "$F/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ -z "$name" ] && continue
  [ $DEST = defined ] && ! grep -qx "$name" <(grep -v '^#' "$F/defined.list" | cut -f1) && continue
  [ "$args" = "-" ] && args=""
  work=$(mktemp -d)
  cp "$F"/*.mrc "$F"/*.mod "$F"/*.prexg "$F"/*.tlt "$work"/
  ls -A "$work" > "$work/.inputs"
  set +e
  (cd "$work" && timeout 300 $PROG $args 2>/dev/null | cat > "$work/.stdout"; echo "${PIPESTATUS[0]}" > "$work/.rc")
  set -e
  mkdir -p "$F/$DEST/$name"
  cp "$work/.stdout" "$F/$DEST/$name.out"
  cp "$work/.rc" "$F/$DEST/$name.rc"
  for f in "$work"/*; do
    b=$(basename "$f")
    grep -qx "$b" "$work/.inputs" || cp "$f" "$F/$DEST/$name/"
  done
  rm -rf "$work"
done
