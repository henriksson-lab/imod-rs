#!/bin/bash
# Regenerate fixtures/point2model/ (inputs and golden/) from the native
# reference point2model.
#
# Inputs, written here: seeded point files in the shapes point2model reads
# (x y z; object contour x y z; comma-separated; with labels, contour times,
# point sizes and values; a file whose last line has no newline; broken files
# for the error paths) and img.mrc, a 16x12x3 byte volume with pixel spacing
# 2.5,2.5,5 written by the native raw2mrc and alterheader.  afs.xyzpt is not
# made here: it is the XYZ point file beadtrack wrote in a native autofidseed
# run on the TS_01 prealigned stack (e2e-ts01, 2026-09-27), converted by
# autofidseed with `-values -1 -sphere 11`.  For each row of cases.tsv the
# native program runs in a fresh directory with stdout captured through a
# pipe; golden/<name>.rc is the exit status, .stdout the standard output and
# .mod the output model, when native leaves one.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
F=$(cd "$(dirname "$0")/point2model" && pwd)
export AUTODOC_DIR=$R/autodoc LD_LIBRARY_PATH=$R/buildlib
S=$(mktemp -d)
cd "$S"
python3 - <<'PY'
import random
r = random.Random(5)
L = ["%.2f %.2f %.1f" % (r.uniform(0, 300), r.uniform(0, 300), i // 4) for i in range(12)]
open('c3.txt', 'w').write('\n'.join(L) + '\n')
L = []
for ob in (1, 2):
    for co in (1, 2, 3):
        for p in range(3):
            L.append("%d %d %.3f %.3f %.3f" % (ob, co, r.uniform(0, 200), r.uniform(0, 200), p + co))
open('c5.txt', 'w').write('\n'.join(L) + '\n')
open('comma.txt', 'w').write('\n'.join(l.replace(' ', ',') for l in L) + '\n')
L = []
for co in (1, 2):
    for p in range(3):
        L.append("1 %d %.2f %.2f %d  label %d-%d here" % (co, r.uniform(0, 99), r.uniform(0, 99), p, co, p))
open('lab.txt', 'w').write('\n'.join(L) + '\n')
L = []
for co in (1, 2, 3):
    for p in range(2):
        if p == 0:
            L.append("1 %d %.2f %.2f %d %d" % (co, r.uniform(0, 99), r.uniform(0, 99), p, co + 4))
        else:
            L.append("1 %d %.2f %.2f %d" % (co, r.uniform(0, 99), r.uniform(0, 99), p))
open('t.txt', 'w').write('\n'.join(L) + '\n')
L = ["1 1 %.2f %.2f %d %.1f %.3f" % (r.uniform(0, 99), r.uniform(0, 99), p, [11, 3.5, 11][p % 3],
                                     r.uniform(0, 1)) for p in range(5)]
open('szv.txt', 'w').write('\n'.join(L) + '\n')
open('skip.txt', 'w').write('# header\ncomment line\n' + open('c3.txt').read())
open('nonl.txt', 'w').write("1 2 3\n4 5 6\n\n7 8 9")
open('bad.txt', 'w').write("1 1 2 3 4\n1 1 2 3\n")
open('badobj.txt', 'w').write("1 1 2 3 4\n0 1 2 3 4\n")
open('one.txt', 'w').write("5\n")
open('img.raw', 'wb').write(bytes(16 * 12 * 3))
PY
$R/mrc/raw2mrc -x 16 -y 12 -z 3 -t byte img.raw img.mrc > /dev/null
$R/flib/image/alterheader -del 2.5,2.5,5 img.mrc > /dev/null
rm -f "$F"/*.txt "$F"/img.mrc
cp *.txt img.mrc "$F"/
rm -rf "$F/golden"; mkdir "$F/golden"
sed "${FULL:+s/^#full\t//;}/^#/d" "$F/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ "$args" = "-" ] && args=""
  d="$S/run/$name"; mkdir -p "$d"
  cp "$F"/*.txt "$F"/*.xyzpt "$F"/img.mrc "$d"/
  set +e
  (cd "$d" && timeout 120 $R/imodutil/point2model $args 2>/dev/null | cat > "$S/stdout"; exit ${PIPESTATUS[0]})
  echo $? > "$F/golden/$name.rc"; set -e
  cp "$S/stdout" "$F/golden/$name.stdout"
  if [ -f "$d/o.mod" ]; then cp "$d/o.mod" "$F/golden/$name.mod"; fi
done
rm -rf "$S"
