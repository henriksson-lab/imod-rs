#!/bin/bash
# Regenerate fixtures/restrictalign/ (inputs and golden/) from the native
# Python restrictalign (IMOD/pysrc/restrictalign, run from the reference
# install) driving the native tiltalign, imodinfo, submfg, xfproduct, ...
#
# Inputs, made here except pt.fid:
#  b1/b3/b8/b25.fid  seeded synthetic fiducial models (1, 3, 8, 25 beads;
#                    written by the native point2model): random 3D points
#                    projected at ts.rawtlt's angles (-51..51 step 3) about a
#                    -90 deg tilt axis with small per-view rotation and mag
#                    changes, 0.4 px noise, 5-10% of points missing in b8/b25
#  pt.fid            the chopped patch-tracking model of the TS_01 e2e
#                    pipeline (pipe/nat/06_xcorr_pt/ts.fid, LengthAndOverlap
#                    16,4), for the unchopped-contour count
#  align.tmpl        copytomocoms' align.com for TS_01 (e2e pipe/nat/23_ctc),
#                    with ModelFile MODEL, ImageFile stub.mrc,
#                    #ImageSizeXandY 512,512, SurfacesToAnalyze 1
#  stub.mrc          a header-only 512 x 512 x 35 MRC (tiltalign and
#                    getmrcsize read only its header)
#  ts.prexg          unit transforms, for align.com's xfproduct step
# Each cases.tsv row is name<TAB>model<TAB>sed script applied to align.tmpl
# (MODEL replaced first; - for none)<TAB>arguments.  A case runs in a fresh
# directory holding every input and align.com; golden/<case>.rc is the exit
# status, .out the standard output (through a pipe), and golden/<case>/ every
# file that is new or changed afterwards.
#
# `make-restrictalign-goldens.sh defined` writes defined/ instead, from our
# own build ($RSBIN, default target/release/imod) for the cases in
# defined.list, whose output an upstream-bug fix changes (BUGS.md).
set -e
REF=${REF:-/tmp/imod-reference-build}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HERE=$ROOT/fixtures/restrictalign
DEST=golden
BIN=$(mktemp -d)
if [ "$1" = defined ]; then
  DEST=defined
  RSBIN=${RSBIN:-$ROOT/target/release/imod}
  mkdir -p $BIN/bin
  for c in restrictalign tiltalign imodinfo submfg xfproduct b3dcopy patch2imod header; do
    ln -s $RSBIN $BIN/bin/$c
  done
  export IMOD_DIR=$BIN AUTODOC_DIR=$ROOT/IMOD/autodoc
else
  # A native install: every reference program and Python script, and pylib
  mkdir -p $BIN/bin; ln -s $REF/pysrc $BIN/pylib
  for d in flib/image flib/model flib/tiltalign imodutil pysrc scripts; do
    for f in $REF/$d/*; do
      n=$(basename $f)
      [ -f $f ] && [ -x $f ] && [[ $n != *.* ]] && [ ! -e $BIN/bin/$n ] && ln -s $f $BIN/bin/$n
    done
  done
  export IMOD_DIR=$BIN AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib
  cd $HERE
  python3 - <<'PY'
import numpy as np, math
r = np.random.default_rng(7)
angles = [-51 + 3 * i for i in range(35)]
open('ts.rawtlt', 'w').write(''.join('%7.2f\n' % a for a in angles))
open('ts.prexg', 'w').write(''.join(
    '   1.0000000   0.0000000   0.0000000   1.0000000       0.000       0.000\n' for a in angles))
def make(name, nbeads, noise=0.4, missing=0.0):
    L = []
    pts = [(r.uniform(-200, 200), r.uniform(-200, 200), r.uniform(-40, 40)) for _ in range(nbeads)]
    for c, (x, y, z) in enumerate(pts):
        for v, a in enumerate(angles):
            if missing and r.uniform() < missing:
                continue
            t = math.radians(a)
            xp = x * math.cos(t) + z * math.sin(t)
            mag = 1 + 0.002 * math.sin(v)
            rot = math.radians(-90 + 0.3 * math.sin(v * 0.7))
            u, w = xp * mag, y * mag
            X = 256 + u * math.cos(rot) - w * math.sin(rot) + r.normal(0, noise)
            Y = 256 + u * math.sin(rot) + w * math.cos(rot) + r.normal(0, noise)
            L.append('1 %d %.2f %.2f %d' % (c + 1, X, Y, v))
    return L
b3 = make('b3', 3)
open('b3.txt', 'w').write('\n'.join(b3) + '\n')
open('b1.txt', 'w').write('\n'.join(b3[:35]) + '\n')
open('b8.txt', 'w').write('\n'.join(make('b8', 8, missing=0.1)) + '\n')
open('b25.txt', 'w').write('\n'.join(make('b25', 25, missing=0.05)) + '\n')
import struct
h = bytearray(1024)
struct.pack_into('<3i', h, 0, 512, 512, 35); struct.pack_into('<i', h, 12, 2)
struct.pack_into('<3i', h, 28, 512, 512, 35); struct.pack_into('<3f', h, 40, 512, 512, 35)
struct.pack_into('<3f', h, 52, 90, 90, 90); struct.pack_into('<3i', h, 64, 1, 2, 3)
h[208:212] = b'MAP '; h[212:216] = bytes([0x44, 0x44, 0, 0])
open('stub.mrc', 'wb').write(h)
PY
  for b in b1 b3 b8 b25; do
    $REF/imodutil/point2model -volume 512,512,35 $b.txt $b.fid > /dev/null
    rm -f $b.txt $b.fid~
  done
fi
rm -rf "$HERE/$DEST"; mkdir -p "$HERE/$DEST"
export PATH=$BIN/bin:$PATH OMP_NUM_THREADS=1
sed "${FULL:+s/^#full\t//;}/^#/d" "$HERE/cases.tsv" | while IFS=$'\t' read -r name model script args; do
  [ -z "$name" ] && continue
  [ $DEST = defined ] && ! grep -qx "$name" <(grep -v '^#' "$HERE/defined.list" | cut -f1) && continue
  [ "$script" = - ] && script=""
  work=$(mktemp -d)
  cp "$HERE"/*.fid "$HERE"/ts.rawtlt "$HERE"/ts.prexg "$HERE"/stub.mrc "$work"/
  sed "s/MODEL/$model/;$script" "$HERE/align.tmpl" > "$work/align.com"
  (cd "$work" && for f in *; do sha256sum "$f"; done) > "$work/.inputs"
  set +e
  (cd "$work" && timeout 900 restrictalign $args 2>/dev/null | cat > "$work/.stdout";
   echo "${PIPESTATUS[0]}" > "$work/.rc")
  set -e
  mkdir -p "$HERE/$DEST/$name"
  cp "$work/.stdout" "$HERE/$DEST/$name.out"
  cp "$work/.rc" "$HERE/$DEST/$name.rc"
  for f in "$work"/*; do
    b=$(basename "$f")
    grep -qx "$(cd $work && sha256sum "$b")" "$work/.inputs" || cp "$f" "$HERE/$DEST/$name/"
  done
  rm -rf "$work"
done
rm -rf $BIN
