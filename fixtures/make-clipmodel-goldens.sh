#!/bin/bash
# Regenerate fixtures/clipmodel/ (inputs and golden/) from the native reference.
# Inputs are authored here, seeded, with native point2model: cl.mod (closed
# contours in three objects, point values, a 2.5 A pixel size so areas and
# cut lengths come out in microns; object 3 is a pair of boundary
# rectangles), op.mod (open contours with point values) and sc.mod (one
# scattered-point contour over five Z values, the shape of the imodfindbeads
# peak model autofidseed clips).  For each row of cases.tsv the native program
# runs in a fresh directory with stdout captured through a pipe and stdin
# empty; golden/<name>.rc is the exit status, .stdout the standard output,
# .mod the output model and .txt the point file, when native leaves one.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
F=$(cd "$(dirname "$0")/clipmodel" && pwd)
export AUTODOC_DIR=$R/autodoc LD_LIBRARY_PATH=$R/buildlib
S=$(mktemp -d)
cd "$S"
python3 - <<'PY'
import math, random
r = random.Random(5)
L = []
c = 0
for z in range(5):
    for k in range(2):
        c += 1
        cx, cy, rad = 20 + 40 * k, 30 + 5 * z, 8 + 2 * k
        for i in range(12):
            a = 2 * math.pi * i / 12
            L.append(f"1 {c} {cx+rad*math.cos(a):.2f} {cy+rad*math.sin(a):.2f} {z} {r.uniform(0,1):.3f}")
c = 0
for z in range(0, 5, 2):
    c += 1
    for (x, y) in [(50, 50), (75, 52), (78, 80), (52, 78), (45, 65)]:
        L.append(f"2 {c} {x+z} {y} {z} {r.uniform(0,1):.3f}")
c = 0
for z in (1, 3):
    c += 1
    for (x, y) in [(10, 10), (70, 10), (70, 70), (10, 70)]:
        L.append(f"3 {c} {x+3*z} {y} {z} 0.5")
open('cl.txt', 'w').write('\n'.join(L) + '\n')
L = []
for c in range(1, 4):
    for i in range(10):
        L.append(f"1 {c} {5+9*i} {20*c+r.uniform(-3,3):.2f} {c-1} {r.uniform(0,1):.3f}")
for c in range(1, 3):
    for i in range(6):
        L.append(f"2 {c} {90-6*i} {15*c} {i%3} {r.uniform(0,1):.3f}")
open('op.txt', 'w').write('\n'.join(L) + '\n')
L = []
for i in range(40):
    L.append(f"1 1 {r.uniform(0,100):.2f} {r.uniform(0,100):.2f} {i%5} {r.uniform(0,1):.3f}")
open('sc.txt', 'w').write('\n'.join(L) + '\n')
PY
$R/imodutil/point2model -values 1 -pixel 2.5,2.5,2.5 -volume 100,100,5 cl.txt cl.mod > /dev/null
$R/imodutil/point2model -open -values 1 -volume 100,100,5 op.txt op.mod > /dev/null
$R/imodutil/point2model -scat -values 1 -volume 100,100,5 sc.txt sc.mod > /dev/null
rm -f "$F"/*.mod
cp cl.mod op.mod sc.mod "$F"/
rm -rf "$F/golden"; mkdir "$F/golden"
sed "${FULL:+s/^#full\t//;}/^#/d" "$F/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ "$args" = "-" ] && args=""
  d="$S/run/$name"; mkdir -p "$d"
  cp "$F"/*.mod "$d"/
  set +e
  (cd "$d" && timeout 120 $R/flib/model/clipmodel $args 2>/dev/null < /dev/null | cat > "$S/stdout"; exit ${PIPESTATUS[0]})
  echo $? > "$F/golden/$name.rc"; set -e
  cp "$S/stdout" "$F/golden/$name.stdout"
  [ -f "$d/o.mod" ] && cp "$d/o.mod" "$F/golden/$name.mod"
  [ -f "$d/o.txt" ] && cp "$d/o.txt" "$F/golden/$name.txt"
done
rm -rf "$S"
