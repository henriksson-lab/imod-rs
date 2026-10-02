#!/bin/bash
# Regenerate fixtures/imodchopconts/ (input and golden/) from the native reference.
# The input col.mod is authored here, seeded: native point2model makes two
# objects of open contours (4 x 35 and 3 x 20 points, one point per Z) from
# text, and the Python below rewrites that file to add the fine-grained data
# imodchopconts carries along: surface numbers on the contours, per-point
# colour changes (with a revert) in each contour's COST chunk, and in each
# object's OBST chunk a per-contour VALUE1, a per-contour TRANS, a per-surface
# TRANS and an object MINMAX1.  For each row of cases.tsv the native program
# runs in a fresh directory with stdout captured through a pipe;
# golden/<name>.rc is the exit status, .stdout the standard output, and .mod
# the output model when native writes one.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
F=$(cd "$(dirname "$0")/imodchopconts" && pwd)
export AUTODOC_DIR=$R/autodoc LD_LIBRARY_PATH=$R/buildlib
S=$(mktemp -d)
cd "$S"
python3 - <<'PY'
import numpy as np
r = np.random.default_rng(7)
L = []
for ob, (nc, npt) in enumerate([(4, 35), (3, 20)], 1):
    for c in range(1, nc + 1):
        x0, y0 = r.uniform(50, 450, 2)
        for z in range(npt):
            L.append('%d %d %.2f %.2f %d' % (ob, c, x0 + r.normal(0, 2) + z, y0 + r.normal(0, 2), z))
open('pts.txt', 'w').write('\n'.join(L) + '\n')
PY
$R/imodutil/point2model -open -zscale 1 pts.txt base.mod > /dev/null
python3 - <<'PY'
import struct
b = open('base.mod', 'rb').read()
out = bytearray(b[:8 + 232]); p = 8 + 232
ob = -1; co = 0
def item(t, f, idx, val):
    return struct.pack('>hH', t, f) + idx + val
def I(i): return struct.pack('>i', i)
def F(x): return struct.pack('>f', x)
COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
while True:
    cid = b[p:p+4]; p += 4
    if cid == b'OBJT':
        ob += 1; co = 0
        out += cid + b[p:p+176]; p += 176
    elif cid == b'CONT':
        psize, flags, time, surf = struct.unpack('>4i', b[p:p+16])
        surf = co % 2 + (1 if ob == 0 else 0)
        out += cid + struct.pack('>4i', psize, flags, time, surf) + b[p+16:p+16+12*psize]
        p += 16 + 12 * psize
        its = b''
        if co % 3 == 0:
            its += item(1, 3 << 2, I(0), bytes(COLORS[2]) + b'\0')
        its += item(1, 3 << 2, I(psize // 3), bytes(COLORS[co % 2]) + b'\0')
        if co % 2 == 1:
            its += item(1, (3 << 2) | (1 << 5), I(2 * psize // 3), b'\0\0\0\0')
        out += b'COST' + I(len(its)) + its
        co += 1
    elif cid == b'IEOF':
        break
    else:
        n = struct.unpack('>i', b[p:p+4])[0]
        chunk = b[p+4:p+4+n]; p += 4 + n
        out += cid + I(n) + chunk
        if cid == b'IMAT':
            its = item(10, 1 << 2, I(0), F(3.5)) + item(3, 0, I(1), I(40)) \
                + item(3, 1 << 6, I(1), I(60)) + item(11, (1 << 4) | (1 << 2) | 1, F(0.5), F(7.0))
            out += b'OBST' + I(len(its)) + its
out += b'IEOF'
open('col.mod', 'wb').write(out)
PY
cp col.mod "$F"/
rm -rf "$F/golden"; mkdir "$F/golden"
sed "${FULL:+s/^#full\t//;}/^#/d" "$F/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ "$args" = "-" ] && args=""
  d="$S/run/$name"; mkdir -p "$d"
  cp "$F"/col.mod "$d"/
  set +e
  (cd "$d" && timeout 120 $R/imodutil/imodchopconts $args 2>/dev/null | cat > "$S/stdout"; exit ${PIPESTATUS[0]})
  echo $? > "$F/golden/$name.rc"; set -e
  cp "$S/stdout" "$F/golden/$name.stdout"
  [ -f "$d/o.mod" ] && cp "$d/o.mod" "$F/golden/$name.mod"
done
rm -rf "$S"
