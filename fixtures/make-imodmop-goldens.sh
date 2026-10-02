#!/bin/bash
# Regenerate fixtures/imodmop/ (inputs and golden/) from the native reference.
# Inputs are authored here, seeded: img.mrc, a 100 x 100 x 5 byte image
# (noise around 110 with dark gold-bead-like disks) written by native raw2mrc,
# and models written by native point2model -- cl.mod (closed contours in
# three objects), op.mod (open contours, for tubes) and scs.mod (scattered
# points with sphere size 4).  For each row of cases.tsv the native program
# runs in a fresh directory with stdout captured through a pipe; native
# xyzproj is put first on PATH for the -project row, since imodmop runs it
# through system().  golden/<name>.rc is the exit status, .stdout the standard
# output and .mrc the output image, when native leaves one.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
F=$(cd "$(dirname "$0")/imodmop" && pwd)
export AUTODOC_DIR=$R/autodoc LD_LIBRARY_PATH=$R/buildlib PATH=$R/flib/image:$PATH
S=$(mktemp -d)
cd "$S"
python3 - <<'PY'
import math, random
import numpy as np
g = np.random.default_rng(7)
v = g.normal(110, 12, (5, 100, 100))
yy, xx = np.mgrid[0:100, 0:100]
for z in range(5):
    for (x, y) in [(22, 30), (61, 44), (48, 77), (80, 18), (15, 85)]:
        v[z][(xx - x - z) ** 2 + (yy - y) ** 2 < 25] -= 70
np.clip(v, 0, 255).astype(np.uint8).tofile('img.raw')
r = random.Random(5)
L = []
c = 0
for z in range(5):
    for k in range(2):
        c += 1
        cx, cy, rad = 20 + 40 * k, 30 + 5 * z, 8 + 2 * k
        for i in range(12):
            a = 2 * math.pi * i / 12
            L.append(f"1 {c} {cx+rad*math.cos(a):.2f} {cy+rad*math.sin(a):.2f} {z}")
c = 0
for z in range(0, 5, 2):
    c += 1
    for (x, y) in [(50, 50), (75, 52), (78, 80), (52, 78), (45, 65)]:
        L.append(f"2 {c} {x+z} {y} {z}")
c = 0
for z in (1, 3):
    c += 1
    for (x, y) in [(10, 10), (70, 10), (70, 70), (10, 70)]:
        L.append(f"3 {c} {x+3*z} {y} {z}")
open('cl.txt', 'w').write('\n'.join(L) + '\n')
L = []
for c in range(1, 4):
    for i in range(10):
        L.append(f"1 {c} {5+9*i} {20*c+r.uniform(-3,3):.2f} {c-1}")
for c in range(1, 3):
    for i in range(6):
        L.append(f"2 {c} {90-6*i} {15*c} {i%3}")
open('op.txt', 'w').write('\n'.join(L) + '\n')
L = []
for i in range(40):
    L.append(f"1 1 {r.uniform(0,100):.2f} {r.uniform(0,100):.2f} {i%5}")
open('sc.txt', 'w').write('\n'.join(L) + '\n')
PY
$R/mrc/raw2mrc -x 100 -y 100 -z 5 -t byte img.raw img.mrc > /dev/null
$R/imodutil/point2model -volume 100,100,5 cl.txt cl.mod > /dev/null
$R/imodutil/point2model -open -volume 100,100,5 op.txt op.mod > /dev/null
$R/imodutil/point2model -scat -sphere 4 -volume 100,100,5 sc.txt scs.mod > /dev/null
rm -f "$F"/*.mod "$F"/*.mrc
cp cl.mod op.mod scs.mod img.mrc "$F"/
rm -rf "$F/golden"; mkdir "$F/golden"
sed "${FULL:+s/^#full\t//;}/^#/d" "$F/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ "$args" = "-" ] && args=""
  d="$S/run/$name"; mkdir -p "$d"
  cp "$F"/*.mod "$F"/img.mrc "$d"/
  set +e
  (cd "$d" && timeout 120 $R/imodutil/imodmop $args 2>/dev/null < /dev/null | cat > "$S/stdout"; exit ${PIPESTATUS[0]})
  echo $? > "$F/golden/$name.rc"; set -e
  cp "$S/stdout" "$F/golden/$name.stdout"
  [ -f "$d/o.mrc" ] && cp "$d/o.mrc" "$F/golden/$name.mrc"
done
rm -rf "$S"
