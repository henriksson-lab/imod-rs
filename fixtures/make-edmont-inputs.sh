#!/bin/bash
# Authors the fixtures/edmont inputs with the native reference raw2mrc and point2model:
# m.st, a 3x2 montage of 16x12 pieces (overlap 4x3), 2 sections, shorts, with its
# piece list m.pl; mh.st the same with the piece coordinates in a SerialEM-style
# extended header (6-byte shorts, flags 2); mm.st with them in mm.st.mdoc; f.st a
# float copy; n2.pl the montage shifted one frame in X; ms.st with sampling (mxyz)
# twice the size and the cell scaled to keep pixel size 2 (BUGS.md, edmont); ex.mod
# an exclusion model with one point on each section.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
export LD_LIBRARY_PATH=$R/buildlib
F=$(cd "$(dirname "$0")" && pwd)/edmont
S=$(mktemp -d); cd "$S"
python3 - <<'PY'
import numpy as np
rng = np.random.default_rng(7)
pl = [(12 * ix, 9 * iy, z) for z in range(2) for iy in range(2) for ix in range(3)]
a = rng.integers(100, 900, size=(len(pl), 12, 16)).astype('<i2')
a.tofile('m.raw')
(a.astype('<f4') * 0.37 - 50).tofile('f.raw')
open('m.pl', 'w').write(''.join('%d %d %d\n' % p for p in pl))
open('n2.pl', 'w').write(''.join('%d %d %d\n' % (x + 12, y, z) for (x, y, z) in pl))
open('pts.txt', 'w').write('1 1 18 5 0\n1 1 5 14 1\n')
PY
$R/mrc/raw2mrc -x 16 -y 12 -z 12 -t short m.raw m.st > /dev/null
$R/mrc/raw2mrc -x 16 -y 12 -z 12 -t float f.raw f.st > /dev/null
python3 - <<'PY'
import struct
pl = [tuple(map(int, l.split())) for l in open('m.pl')]
d = bytearray(open('m.st', 'rb').read())
ext = b''.join(struct.pack('<3h', *p) for p in pl)
hdr = bytearray(d[:1024])
struct.pack_into('<i', hdr, 92, len(ext))
struct.pack_into('<hh', hdr, 128, 6, 2)
open('mh.st', 'wb').write(bytes(hdr) + ext + bytes(d[1024:]))
open('mm.st', 'wb').write(d)
with open('mm.st.mdoc', 'w') as f:
    f.write('PixelSpacing = 2.5\nImageFile = mm.st\nImageSize = 16 12\nMontage = 1\n'
            'DataMode = 1\n\n')
    for i, p in enumerate(pl):
        f.write('[ZValue = %d]\nPieceCoordinates = %d %d %d\nTiltAngle = %g\n\n'
                % (i, *p, 3.0 * i))
s = bytearray(d)
struct.pack_into('<3i', s, 28, 32, 24, 12)
struct.pack_into('<3f', s, 40, 64.0, 48.0, 24.0)
open('ms.st', 'wb').write(s)
PY
$R/imodutil/point2model -input pts.txt -output ex.mod -image m.st > /dev/null
cp m.st mh.st mm.st mm.st.mdoc f.st ms.st m.pl n2.pl ex.mod "$F/"
rm -rf "$S"
