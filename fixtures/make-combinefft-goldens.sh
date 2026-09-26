#!/bin/bash
# Regenerate fixtures/combinefft/ (inputs and golden/) from the native reference.
# Inputs are authored here: two seeded synthetic 15x13x7 float volumes (a
# Gaussian-blob phantom plus noise, B shifted and scaled) written by native
# raw2mrc, short and byte copies made by native newstack, and their 3-D FFTs
# made the way volcombine does (native taperoutvol, then fftrans -3dfft).
# Tilt-angle files and inverse transforms are plain text.  For each row of
# cases.tsv the native program runs in a fresh directory with stdout captured
# through a pipe; golden/<name>.rc is the exit status, .stdout the standard
# output, and .out the output file (out.fft / out.mrc, or bb.fft when the
# output is written into the second input), when native leaves one.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
F=$(cd "$(dirname "$0")/combinefft" && pwd)
export AUTODOC_DIR=$R/autodoc LD_LIBRARY_PATH=$R/buildlib
S=$(mktemp -d)
cd "$S"
python3 - <<'PY'
import numpy as np
def phantom(nx, ny, nz, seed):
    r = np.random.default_rng(seed)
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32)
    v = np.zeros((nz, ny, nx), np.float32)
    for k in range(6):
        cx, cy, cz = r.uniform(0, nx), r.uniform(0, ny), r.uniform(0, nz)
        s = r.uniform(1.2, 3)
        v += r.uniform(20, 80) * np.exp(-((x-cx)**2 + (y-cy)**2 + (z-cz)**2) / (2*s*s))
    return v
rng = np.random.default_rng(5)
p = phantom(15, 13, 7, 1)
(p + rng.normal(0, 3, p.shape) + 100).astype(np.float32).tofile('a.raw')
(np.roll(p, 1, axis=2) * 0.97 + rng.normal(0, 3, p.shape) + 95).astype(np.float32).tofile('b.raw')
open('a.tlt', 'w').write(''.join(f'{a:8.2f}\n' for a in np.arange(-60, 60.1, 3)))
open('b.tlt', 'w').write(''.join(f'{a:8.2f}\n' for a in np.arange(-57, 63.1, 3)))
ang = [-66.1, -61.2, -55.9, -50.3, -44.7, -38.1, -31.6, -24.2, -17.5, -10.1, -3.3, 2.9,
       9.8, 16.4, 23.8, 30.2, 37.7, 43.3, 50.1, 55.2, 61.4, 64.9]
open('c.tlt', 'w').write(''.join(f'{a}\n' for a in ang[::-1]))
open('one.tlt', 'w').write('5.0\n')
open('bad.tlt', 'w').write('1.0\n2.0\nabc\n')
open('inv.xf', 'w').write(' 0.0213 0.9995 0.0105 5.2\n-0.9991 0.0198 0.0031 -2.1\n 0.0102 -0.0035 1.0012 0.5\n')
open('short.xf', 'w').write('1 0 0\n0 1 0\n')
open('bad.xf', 'w').write('1 0 x\n0 1 0\n0 0 1\n')
PY
$R/mrc/raw2mrc -x 15 -y 13 -z 7 -t float a.raw a.mrc > /dev/null
$R/mrc/raw2mrc -x 15 -y 13 -z 7 -t float b.raw b.mrc > /dev/null
$R/flib/image/newstack -mode 1 -scale 0,30000 a.mrc as.mrc > /dev/null
$R/flib/image/newstack -mode 0 -scale 0,255 b.mrc bb.mrc > /dev/null
for s in a b; do
  $R/flib/image/taperoutvol -input $s.mrc -output ${s}_tap.mrc -taper 2,2,1 > /dev/null
  $R/flib/image/fftrans -3dfft ${s}_tap.mrc $s.fft > /dev/null
done
rm -f "$F"/*.mrc "$F"/*.fft "$F"/*.tlt "$F"/*.xf
cp a.mrc b.mrc as.mrc bb.mrc a.fft b.fft *.tlt *.xf "$F"/
rm -rf "$F/golden"; mkdir "$F/golden"
sed "${FULL:+s/^#full\t//;}/^#/d" "$F/cases.tsv" | while IFS=$'\t' read -r name args stdin; do
  [ "$args" = "-" ] && args=""
  [ "$stdin" = "-" ] && stdin=""
  d="$S/run/$name"; mkdir -p "$d"
  cp "$F"/*.mrc "$F"/*.fft "$F"/*.tlt "$F"/*.xf "$d"/
  cp "$d/b.fft" "$d/bb.fft"
  set +e
  (cd "$d" && printf -- "$stdin" | OMP_NUM_THREADS=1 timeout 120 \
      $R/flib/image/combinefft $args 2>/dev/null | cat > "$S/stdout"; exit ${PIPESTATUS[1]})
  echo $? > "$F/golden/$name.rc"; set -e
  cp "$S/stdout" "$F/golden/$name.stdout"
  for o in out.fft out.mrc; do [ -f "$d/$o" ] && cp "$d/$o" "$F/golden/$name.out"; done
  cmp -s "$d/bb.fft" "$F/b.fft" || cp "$d/bb.fft" "$F/golden/$name.out"
done
rm -rf "$S"
# Defined behaviour where native has a bug (BUGS.md, "combinefft"): the
# translation spells the "You must enter" message with its missing blank.
sed -i 's/You mustenter/You must enter/' "$F/golden/e_no_tilts.stdout"
