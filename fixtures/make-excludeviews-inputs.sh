#!/bin/bash
# Authors fixtures/excludeviews/inputs with the native reference raw2mrc and the
# native excludeviews (IMOD/pysrc/excludeviews with native programs on PATH):
#   ts.st (16x12x6 shorts) with ts.rawtlt and ts.st.mdoc (exposure and prior
#   record doses); tsw.st/tsw.st.mdoc the same with one PriorRecordDose missing;
#   plain.st a copy of ts.st with nothing else; mt.st and mp.st/mp.pl the
#   fixtures/edmont montages with header and piece-list coordinates.
#   s1_*: ts after `excludeviews -views 2,4-5 ts.st`; s5_*: s1 after a second
#   `-views 1`; s2_*: mp after `-views 2 -mont`; s3_*: plain after
#   `-views 1,6 -del`; s4_*: mt after `-views 2`.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
F=$ROOT/fixtures/excludeviews/inputs
NAT=${NATIMOD:-/big/henriksson/extwork/natimod}
export LD_LIBRARY_PATH=$R/buildlib AUTODOC_DIR=$R/autodoc
S=$(mktemp -d); cd "$S"
python3 - <<'PY'
import numpy as np
rng = np.random.default_rng(3)
rng.integers(0, 1000, size=(6, 12, 16)).astype('<i2').tofile('s.raw')
open('ts.rawtlt', 'w').write(''.join('%.2f\n' % a for a in [-30, -20, -10, 0, 10, 20]))
def mdoc(name, skip):
    with open(name + '.mdoc', 'w') as f:
        f.write('PixelSpacing = 2.5\nImageFile = %s\nImageSize = 16 12\nDataMode = 1\n\n' % name)
        for i in range(6):
            f.write('[ZValue = %d]\nTiltAngle = %g\nExposureDose = 1.5\n' % (i, -30 + 10 * i))
            if i != skip:
                f.write('PriorRecordDose = %g\n' % (1.5 * i))
            f.write('\n')
mdoc('ts.st', -1)
mdoc('tsw.st', 2)
PY
$R/mrc/raw2mrc -x 16 -y 12 -z 6 -t short s.raw ts.st > /dev/null
rm s.raw
cp ts.st plain.st; cp ts.st tsw.st
cp "$ROOT/fixtures/edmont/mh.st" mt.st
cp "$ROOT/fixtures/edmont/m.st" mp.st; cp "$ROOT/fixtures/edmont/m.pl" mp.pl
cp ts.st ts.rawtlt ts.st.mdoc tsw.st tsw.st.mdoc plain.st mt.st mp.st mp.pl "$F/"
stage() {  # stage <prefix> <excludeviews args...>: run natively in the current dir
  p=$1; shift
  env IMOD_DIR=$NAT PATH=$NAT/bin:$PATH PYTHONPATH=$ROOT/IMOD/pysrc \
      python3 "$ROOT/IMOD/pysrc/excludeviews" "$@" > /dev/null
  for f in *; do cp "$f" "$F/${p}_$f"; done
}
mkdir s1 && cp ts.st ts.rawtlt ts.st.mdoc s1/ && (cd s1 && stage s1 -views 2,4-5 ts.st)
cp -a s1 s5 && (cd s5 && stage s5 -views 1 ts.st)
mkdir s2 && cp mp.st mp.pl s2/ && (cd s2 && stage s2 -views 2 -mont mp.st)
mkdir s3 && cp plain.st s3/ && (cd s3 && stage s3 -views 1,6 -del plain.st)
mkdir s4 && cp mt.st s4/ && (cd s4 && stage s4 -views 2 mt.st)
rm -rf "$S"
