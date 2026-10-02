#!/bin/bash
# Regenerate fixtures/sortbeadsurfs/ (inputs and golden/) from the native
# reference.  Inputs are authored here, seeded, in the shape autofidseed feeds
# the program: beadtrack's XYZOutputFile (contour, X, Y, Z, mean residual)
# for 64 beads on two tilted surfaces, a few with outlying residuals, turned
# into xyz.mod by native `point2model -values -1 -sphere 5` (contour values);
# xyzf.mod is the same with Y and Z swapped (a flipped model, found from
# maxz > maxy), and sorted.mod native sortbeadsurfs's own two-object output
# for xyz.mod (input for -already/-check).  For each row of cases.tsv the
# native program runs in a fresh directory with stdout captured through a
# pipe; golden/<name>.rc is the exit status, .stdout the standard output,
# .mod the output model and .txt the surface text file, when native leaves one.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
F=$(cd "$(dirname "$0")/sortbeadsurfs" && pwd)
export AUTODOC_DIR=$R/autodoc LD_LIBRARY_PATH=$R/buildlib
S=$(mktemp -d)
cd "$S"
python3 - <<'PY'
import random
r = random.Random(11)
L = []
c = 0
for s, zb in ((0, 130.), (1, 70.)):
    for i in range(32):
        c += 1
        x = r.uniform(20, 1000); y = r.uniform(20, 1000)
        z = zb + 0.02 * x - 0.01 * y + r.gauss(0, 1.5)
        v = abs(r.gauss(0.3, 0.1)) if i % 9 else 2.5
        L.append(f"{c} {x:.2f} {y:.2f} {z:.2f} {v:.3f}")
r.shuffle(L)
L = [f"{k+1} " + l.split(' ', 1)[1] for k, l in enumerate(L)]
open('xyz.txt', 'w').write('\n'.join(L) + '\n')
F = []
for l in L:
    c, x, y, z, v = l.split()
    F.append(f"{c} {x} {z} {y} {v}")
open('xyzf.txt', 'w').write('\n'.join(F) + '\n')
PY
$R/imodutil/point2model -values -1 -sphere 5 -volume 1024,1024,200 xyz.txt xyz.mod > /dev/null
$R/imodutil/point2model -values -1 -sphere 5 -volume 1024,200,1024 xyzf.txt xyzf.mod > /dev/null
$R/flib/model/sortbeadsurfs xyz.mod sorted.mod > /dev/null
rm -f "$F"/*.mod
cp xyz.mod xyzf.mod sorted.mod "$F"/
rm -rf "$F/golden"; mkdir "$F/golden"
sed "${FULL:+s/^#full\t//;}/^#/d" "$F/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ "$args" = "-" ] && args=""
  d="$S/run/$name"; mkdir -p "$d"
  cp "$F"/*.mod "$d"/
  set +e
  (cd "$d" && timeout 120 $R/flib/model/sortbeadsurfs $args 2>/dev/null < /dev/null | cat > "$S/stdout"; exit ${PIPESTATUS[0]})
  echo $? > "$F/golden/$name.rc"; set -e
  cp "$S/stdout" "$F/golden/$name.stdout"
  [ -f "$d/o.mod" ] && cp "$d/o.mod" "$F/golden/$name.mod"
  [ -f "$d/o.txt" ] && cp "$d/o.txt" "$F/golden/$name.txt"
done
rm -rf "$S"
