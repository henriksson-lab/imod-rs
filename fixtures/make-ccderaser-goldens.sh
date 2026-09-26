#!/bin/bash
# Regenerate fixtures/ccderaser/ (inputs and golden/) from the native reference.
# Inputs are authored here: seeded synthetic images (noise, a gradient, hot
# pixels, 2-3 pixel X-rays, giant peaks, dark beads, a hot column segment and
# edge hot pixels) written by native raw2mrc as float, short and byte MRC, and
# models written by native point2model (patch, line, circle-with-sizes,
# boundary, all-section, montage and error-case objects; mv.mod additionally
# has its object's valblack and matflags2 SKIP_LOW patched in its IMAT chunk
# so -skip has something to skip).  For each row of cases.tsv the native
# program runs in a fresh directory with stdout captured through a pipe;
# golden/<name>.rc is the exit status, .stdout the standard output, and
# .o.mrc/.p.mod/.inplace the output image, point model, or input image
# modified in place, when native leaves one.  Noise fill (-order < 0) is not
# here: it is seeded from time(NULL).
#
# `make-ccderaser-goldens.sh defined` instead writes defined/ from the fixed
# translation ($RSBIN, default target/release/imod) for the cases named in
# defined.list -- those whose output an upstream-bug fix changes (BUGS.md,
# 2026-09-26) -- using the inputs already in place; golden/ is untouched.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
F=$(cd "$(dirname "$0")/ccderaser" && pwd)
if [ "$1" = defined ]; then
  RSBIN=${RSBIN:-$(cd "$F/../.." && pwd)/target/release/imod}
  export AUTODOC_DIR=$(cd "$F/../.." && pwd)/IMOD/autodoc
  rm -rf "$F/defined"; mkdir "$F/defined"
  S=$(mktemp -d)
  grep -v '^#' "$F/defined.list" | cut -f1 | while read -r name; do
    [ -z "$name" ] && continue
    IFS=$'\t' read -r _ args stdin < <(grep -P "^$name\t" "$F/cases.tsv")
    [ "$args" = "-" ] && args=""
    [ "$stdin" = "-" ] && stdin=""
    d="$S/run/$name"; mkdir -p "$d"
    cp "$F"/*.mrc "$F"/*.mod "$F"/mont.pl "$F"/par.txt "$d"/
    set +e
    (cd "$d" && printf -- "$stdin" | OMP_NUM_THREADS=1 timeout 120 \
        "$RSBIN" ccderaser $args 2>/dev/null | cat > "$S/stdout"; exit ${PIPESTATUS[1]})
    echo $? > "$F/defined/$name.rc"; set -e
    cp "$S/stdout" "$F/defined/$name.stdout"
    [ -f "$d/o.mrc" ] && cp "$d/o.mrc" "$F/defined/$name.o.mrc"
    [ -f "$d/p.mod" ] && cp "$d/p.mod" "$F/defined/$name.p.mod"
    for i in f s b; do
      cmp -s "$d/$i.mrc" "$F/$i.mrc" || cp "$d/$i.mrc" "$F/defined/$name.inplace"
    done
  done
  rm -rf "$S"
  exit 0
fi
export AUTODOC_DIR=$R/autodoc LD_LIBRARY_PATH=$R/buildlib
S=$(mktemp -d)
cd "$S"
python3 - <<'PY'
import numpy as np, json, math
def make(nx, ny, nz, mean, sd, seed, feat):
    r = np.random.default_rng(seed)
    a = r.normal(mean, sd, size=(nz, ny, nx)).astype(np.float32)
    yy, xx = np.mgrid[0:ny, 0:nx]
    for z in range(nz):
        a[z] += (xx * 0.3 + yy * 0.2).astype(np.float32)
        for k in range(12):
            x = r.integers(0, nx); y = r.integers(0, ny)
            a[z, y, x] += r.uniform(300, 1500) * (1 if r.random() < 0.8 else -1) * sd / 30
        for k in range(5):
            x = r.integers(3, nx - 4); y = r.integers(3, ny - 4)
            v = r.uniform(400, 1200) * sd / 30
            feat.append(('xray', z, x, y))
            a[z, y, x] += v; a[z, y, x + 1] += v * 0.7
            if r.random() < 0.5: a[z, y + 1, x] += v * 0.5
        x = r.integers(12, nx - 12); y = r.integers(12, ny - 12)
        feat.append(('giant', z, x, y))
        d = np.hypot(xx - x, yy - y)
        a[z] += np.where(d < 5, 900 * sd / 30 * (1 - d / 6), 0).astype(np.float32)
        for k in range(4):
            x = r.uniform(8, nx - 8); y = r.uniform(8, ny - 8)
            feat.append(('bead', z, x, y))
            d = np.hypot(xx + 0.5 - x, yy + 0.5 - y)
            a[z] -= np.where(d < 4.5, 400 * sd / 30 * (1 - (d / 4.5) ** 2), 0).astype(np.float32)
        c = r.integers(10, nx - 10)
        feat.append(('col', z, c, 0))
        a[z, 10:ny - 10, c] += 250 * sd / 30
        a[z, 0, r.integers(0, nx)] += 1200 * sd / 30
        a[z, r.integers(0, ny), nx - 1] += 1200 * sd / 30
    return a
F = []
a = make(72, 56, 2, 1000., 30., 7, F)
a.tofile('f.raw')
np.clip(a, -32768, 32767).astype(np.int16).tofile('s.raw')
np.clip(make(67, 53, 2, 100., 8., 9, []), 0, 255).astype(np.uint8).tofile('b.raw')
L = []; c = 0
for t, z, x, y in F:
    if t == 'xray':
        c += 1; L += [(1, c, x + 0.5, y + 0.5, z, 1), (1, c, x + 1.5, y + 0.5, z, 1)]
c = 0
for t, z, x, y in F:
    if t == 'col':
        c += 1; L += [(2, c, x + 0.5, 12.5, z, 1), (2, c, x + 0.5, 43.5, z, 1)]
for z in range(2):
    L += [(3, z + 1, x, y, z, 4.5) for t, zz, x, y in F if t == 'bead' and zz == z]
c = 0
for t, z, x, y in F:
    if t == 'giant':
        c += 1
        L += [(4, c, x + 0.5 + 6.5 * math.cos(k * math.pi / 3),
               y + 0.5 + 6.5 * math.sin(k * math.pi / 3), z, 1) for k in range(6)]
L += [(5, 1, 30 + 20 * math.cos(k * math.pi / 6), 28 + 20 * math.sin(k * math.pi / 6), 1, 1)
      for k in range(12)]
L += [(6, 1, 40.5, 30.5, 0, 1), (6, 2, 20.5, 40.5, 0, 1), (6, 2, 21.5, 40.5, 0, 1)]
open('m1.txt', 'w').write(''.join('%d %d %.3f %.3f %d %.2f\n' % l for l in L))
L = [(1, z + 1, x, y, z) for z in range(2) for t, zz, x, y in F if t == 'bead' and zz == z]
open('mb.txt', 'w').write(''.join('%d %d %.3f %.3f %d\n' % l for l in L))
L = [(1, 1, x, y, z) for t, z, x, y in F if t == 'bead']
open('mbs.txt', 'w').write(''.join('%d %d %.3f %.3f %d\n' % l for l in L))
L = []; c = 0
for t, z, x, y in F:
    if t == 'xray':
        c += 1; L += [(1, c, x + 0.5, y + 0.5, z, c * 10.), (1, c, x + 1.5, y + 0.5, z, c * 10.)]
open('mv.txt', 'w').write(''.join('%d %d %.3f %.3f %d %.1f\n' % l for l in L))
open('mont.pl', 'w').write('0 0 0\n62 0 1\n')
texts = {
    'mm': '1 1 70.5 40.5 0 1\n1 1 71.5 40.5 0 1\n1 2 100.5 30.5 1 1\n1 3 20.5 20.5 0 1\n',
    'mbad': '1 1 500.5 40.5 0 1\n',
    'mz': '1 1 20.5 20.5 0 1\n1 1 21.5 20.5 1 1\n',
    'ml3': '1 1 20.5 20.5 0 1\n1 1 21.5 20.5 0 1\n1 1 22.5 20.5 0 1\n',
    'mld': '1 1 20.5 20.5 0 1\n1 1 25.5 25.5 0 1\n',
    'mlz': '1 1 20.5 20.5 0 1\n1 1 20.5 25.5 1 1\n',
    'mlines': '1 1 10.5 30.5 0 1\n1 1 60.5 30.5 0 1\n1 2 20.5 31.5 0 1\n1 2 50.5 31.5 0 1\n'
              '1 3 15.5 29.5 0 1\n1 3 40.5 29.5 0 1\n1 4 40.5 10.5 1 1\n1 4 40.5 50.5 1 1\n'
              '1 5 41.5 20.5 1 1\n1 5 41.5 40.5 1 1\n1 6 3.5 5.5 0 1\n1 6 3.5 20.5 0 1\n',
    'mbigc': '1 1 30.5 30.5 0 120\n',
    'mtap': '1 1 -30 -30 1 1\n1 1 100 -30 1 1\n1 1 100 90 1 1\n1 1 -30 90 1 1\n',
    'medge': '1 1 -5 -5 1 1\n1 1 45 -5 1 1\n1 1 45 30 1 1\n1 1 -5 30 1 1\n1 2 50.5 40.5 0 1\n'
             '1 2 51.5 40.5 0 1\n1 3 50 10 0 1\n1 3 65 10 0 1\n1 3 68 25 0 1\n1 3 55 28 0 1\n',
}
for k, v in texts.items():
    open(k + '.txt', 'w').write(v)
open('par.txt', 'w').write('FindPeaks\nPeakCriterion 7\nPointModel p.mod\nVerbose 1\n')
PY
$R/mrc/raw2mrc -x 72 -y 56 -z 2 -t float f.raw f.mrc > /dev/null
$R/mrc/raw2mrc -x 72 -y 56 -z 2 -t short s.raw s.mrc > /dev/null
$R/mrc/raw2mrc -x 67 -y 53 -z 2 -t byte b.raw b.mrc > /dev/null
$R/imodutil/point2model -sizes -image f.mrc m1.txt m1.mod > /dev/null
$R/imodutil/point2model -sphere 5 mb.txt mb.mod > /dev/null
$R/imodutil/point2model -scat -sphere 4 mbs.txt mbs.mod > /dev/null
$R/imodutil/point2model -values -1 -flags 1 mv.txt mv.mod > /dev/null
python3 - <<'PY'
d = bytearray(open('mv.mod', 'rb').read())
k = d.find(b'IMAT') + 8
d[k + 12] = 100      # valblack
d[k + 14] |= 1       # matflags2 |= MATFLAGS2_SKIP_LOW
open('mv.mod', 'wb').write(d)
PY
for m in mm mbad mz ml3 mld mlz mlines mbigc mtap medge; do
  $R/imodutil/point2model -sizes $m.txt $m.mod > /dev/null
done
rm -f "$F"/*.mrc "$F"/*.mod "$F"/*.pl "$F"/par.txt
cp *.mrc *.mod mont.pl par.txt "$F"/
rm -rf "$F/golden"; mkdir "$F/golden"
sed "${FULL:+s/^#full\t//;}/^#/d" "$F/cases.tsv" | while IFS=$'\t' read -r name args stdin; do
  [ "$args" = "-" ] && args=""
  [ "$stdin" = "-" ] && stdin=""
  d="$S/run/$name"; mkdir -p "$d"
  cp "$F"/*.mrc "$F"/*.mod "$F"/mont.pl "$F"/par.txt "$d"/
  set +e
  (cd "$d" && printf -- "$stdin" | OMP_NUM_THREADS=1 timeout 120 \
      $R/flib/model/ccderaser $args 2>/dev/null | cat > "$S/stdout"; exit ${PIPESTATUS[1]})
  echo $? > "$F/golden/$name.rc"; set -e
  cp "$S/stdout" "$F/golden/$name.stdout"
  [ -f "$d/o.mrc" ] && cp "$d/o.mrc" "$F/golden/$name.o.mrc"
  [ -f "$d/p.mod" ] && cp "$d/p.mod" "$F/golden/$name.p.mod"
  for i in f s b; do
    cmp -s "$d/$i.mrc" "$F/$i.mrc" || cp "$d/$i.mrc" "$F/golden/$name.inplace"
  done
done
rm -rf "$S"
