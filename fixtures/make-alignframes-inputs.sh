#!/bin/bash
# Inputs for the alignframes golden suite (fixtures/alignframes/).  Image
# containers are written by the native reference `raw2mrc` from seeded content
# (a smooth random image shifted a few pixels per frame, plus noise), so they
# are IMOD's own files; the extended-header movie gets its extended header
# patched in with Python, as make-etomo-prog-inputs.sh does for piece lists.
# Text inputs (mdoc, tilt, dose, defect, list and frame-list files) are written
# directly.  Run once; the outputs are committed fixtures.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
export LD_LIBRARY_PATH=$R/buildlib AUTODOC_DIR=$R/autodoc
F=$(cd "$(dirname "$0")" && pwd)/alignframes
mkdir -p "$F"
S=$(mktemp -d)
cd "$S"
python3 - <<'EOF'
import numpy as np
r = np.random.default_rng(20261005)

def smooth(n, m, sigma):
    a = r.normal(0, 1, (n, m))
    f = np.fft.fft2(a)
    ky = np.fft.fftfreq(n)[:, None]
    kx = np.fft.fftfreq(m)[None, :]
    f *= np.exp(-(kx * kx + ky * ky) * (2 * np.pi * sigma) ** 2 / 2)
    b = np.real(np.fft.ifft2(f))
    return (b - b.mean()) / b.std()

def movie(nx, ny, nz, shifts, base, amp, noise, big):
    out = np.empty((nz, ny, nx))
    for z in range(nz):
        dx, dy = shifts[z]
        out[z] = base + amp * big[20 + dy:20 + dy + ny, 20 + dx:20 + dx + nx]
        out[z] += r.normal(0, noise, (ny, nx))
    return out

# The main movie: 8 frames of 96 x 96, drifting about 1-2 pixels per frame
big = smooth(140, 140, 3.0)
sh = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 3), (6, 3), (7, 4)]
m = movie(96, 96, 8, sh, 20., 6., 2., big)
m.astype(np.float32).tofile('mov.raw')
np.clip(np.rint(m * 10), 0, 32000).astype(np.int16).tofile('movs.raw')
np.clip(np.rint(m * 0.3), 0, 255).astype(np.uint8).tofile('movb.raw')

# Same movie with pairs of identical frames, for block grouping
mp = np.repeat(m[::2, :64, :64], 2, axis=0)
mp.astype(np.float32).tofile('movp.raw')

# Gain reference near 1, and a dark reference
(1. + 0.05 * r.normal(0, 1, (96, 96))).astype(np.float32).tofile('gainref.raw')
r.integers(0, 6, (96, 96)).astype(np.int16).tofile('dark.raw')

# A tilt series of three short movies, 64 x 64 x 6
big2 = smooth(110, 110, 2.5)
for i in range(3):
    shs = [(k + i % 2, (k * (i + 1)) // 2) for k in range(6)]
    t = movie(64, 64, 6, shs, 15., 5., 2., big2)
    np.clip(np.rint(t * 10), 0, 32000).astype(np.int16).tofile(f'ts{i + 1}.raw')

# Six single-frame files, 64 x 64 float, for -break combining
for i in range(6):
    t = movie(64, 64, 1, [(i, i // 2)], 10., 4., 1.5, big2)
    t.astype(np.float32).tofile(f'one{i + 1}.raw')

# A single-exposure frame tilt series: 30 frames of 48 x 48 in 4 sets
ft = []
for k in range(30):
    ft.append(movie(48, 48, 1, [(k % 7, k % 5)], 12., 4., 1.5, big2)[0])
np.array(ft).astype(np.float32).tofile('fts.raw')

# A movie whose extended header carries tilt angles (UCSF tomo layout):
# 12 frames, 48 x 48, nint 0 nreal 12, three tilts of 4 frames
u = movie(48, 48, 12, [(k % 6, k % 4) for k in range(12)], 10., 4., 1.5, big2)
u.astype(np.float32).tofile('ucsf.raw')

# Block grouping of three frames (BUGS.md, alignframes): 9 frames that are three
# distinct images each repeated three times, in float and byte, and the 3-frame
# movies of the group sums those should align as (3 x each image, in float and
# in short)
g3 = movie(48, 48, 3, [(0, 0), (2, 1), (4, 3)], 10., 4., 1.5, big2).astype(np.float32)
np.repeat(g3, 3, axis=0).tofile('mov9.raw')
(g3 * np.float32(3)).tofile('mov3x.raw')
g3b = np.clip(np.rint(g3 * 8), 0, 255).astype(np.uint8)
np.repeat(g3b, 3, axis=0).tofile('mov9b.raw')
(g3b.astype(np.int16) * 3).tofile('mov3s.raw')
EOF
$R/mrc/raw2mrc -x 96 -y 96 -z 8 -t float mov.raw mov.mrc > /dev/null
$R/mrc/raw2mrc -x 64 -y 64 -z 8 -t float movp.raw movp.mrc > /dev/null
$R/mrc/raw2mrc -x 96 -y 96 -z 8 -t short movs.raw movs.mrc > /dev/null
$R/mrc/raw2mrc -x 96 -y 96 -z 8 -t byte movb.raw movb.mrc > /dev/null
$R/mrc/raw2mrc -x 96 -y 96 -z 1 -t float gainref.raw gainref.mrc > /dev/null
$R/mrc/raw2mrc -x 96 -y 96 -z 1 -t short dark.raw dark.mrc > /dev/null
for i in 1 2 3; do $R/mrc/raw2mrc -x 64 -y 64 -z 6 -t short ts$i.raw ts$i.mrc > /dev/null; done
for i in 1 2 3 4 5 6; do $R/mrc/raw2mrc -x 64 -y 64 -z 1 -t float one$i.raw one$i.mrc > /dev/null; done
$R/mrc/raw2mrc -x 48 -y 48 -z 30 -t float fts.raw fts.mrc > /dev/null
$R/mrc/raw2mrc -x 48 -y 48 -z 12 -t float ucsf.raw ucsf0.mrc > /dev/null
$R/mrc/raw2mrc -x 48 -y 48 -z 9 -t float mov9.raw mov9.mrc > /dev/null
$R/mrc/raw2mrc -x 48 -y 48 -z 3 -t float mov3x.raw mov3x.mrc > /dev/null
$R/mrc/raw2mrc -x 48 -y 48 -z 9 -t byte mov9b.raw mov9b.mrc > /dev/null
$R/mrc/raw2mrc -x 48 -y 48 -z 3 -t short mov3s.raw mov3s.mrc > /dev/null
python3 - <<'EOF'
import struct
# Extended header: nint 0, nreal 12; slot 0 the tilt angle, 10 the axis
# rotation, 11 the pixel size in Angstroms
d = bytearray(open('ucsf0.mrc', 'rb').read())
hdr, data = d[:1024], d[1024:]
ext = b''
for z in range(12):
    vals = [0.0] * 12
    vals[0] = [-20.0, 0.0, 20.0][z // 4]
    vals[10] = 86.3
    vals[11] = 2.5
    ext += struct.pack('<12f', *vals)
struct.pack_into('<i', hdr, 92, len(ext))
struct.pack_into('<hh', hdr, 128, 0, 12)
open('ucsf.mrc', 'wb').write(hdr + ext + data)
EOF
# A stack the tilt-series movies correspond to (sections at binning 2)
python3 -c "
import numpy as np
np.zeros((3, 32, 32), np.int16).tofile('stk.raw')"
$R/mrc/raw2mrc -x 32 -y 32 -z 3 -t short stk.raw tsstack.mrc > /dev/null
# The same stack with a 24-byte extended header (nint 0, nreal 2), not tilt
# angles, for the copy of a stack's extended header to every output
python3 - <<'PYEOF'
import struct
d = bytearray(open('tsstack.mrc', 'rb').read())
hdr, data = d[:1024], d[1024:]
ext = struct.pack('<6f', 1., 2., 3., 4., 5., 6.)
struct.pack_into('<i', hdr, 92, len(ext))
struct.pack_into('<hh', hdr, 128, 0, 2)
open('tsstackx.mrc', 'wb').write(hdr + ext + data)
PYEOF
cat > ts.mdoc <<'EOF'
PixelSpacing = 2.4
DataMode = 1
ImageSize = 64 64
ImageFile = ts.mrc

[T = SerialEM: Digitized on EMBL Krios   binning = 1    02-Oct-26  10:11:12  ]

[T =     Tilt axis angle = 85.6, binning = 1  spot = 8  camera = 0]

[ZValue = 0]
TiltAngle = 0.01
PixelSpacing = 2.4
Binning = 1
ExposureDose = 2.5
SubFramePath = X:\frames\ts1.mrc
MinMaxMean = 10 200 150
DateTime = 02-Oct-26  10:11:12
FrameDosesAndNumber = 0.4 6

[ZValue = 1]
TiltAngle = 3.01
PixelSpacing = 2.4
Binning = 1
ExposureDose = 2.6
SubFramePath = X:\frames\ts2.mrc
MinMaxMean = 10 200 151
DateTime = 02-Oct-26  10:12:12
FrameDosesAndNumber = 0.45 6

[ZValue = 2]
TiltAngle = -3.0
PixelSpacing = 2.4
Binning = 1
ExposureDose = 2.4
SubFramePath = X:\frames\ts3.mrc
MinMaxMean = 10 200 149
DateTime = 02-Oct-26  10:13:12
FrameDosesAndNumber = 0.4 3 0.42 3
EOF
printf '0.01\n3.01\n-3.0\n' > ts.tlt
printf '2.5\n2.6\n2.4\n' > dose1.txt
printf '0 2.5\n2.5 2.6\n5.1 2.4\n' > dose2.txt
printf '0 2.5\n2.5 5.1\n5.1 7.5\n' > dose3.txt
printf '0.4 8\n' > dose5.txt
printf 'ts1.mrc\nts2.mrc\nts3.mrc\n' > list.txt
cat > defects.txt <<'EOF'
CameraSizeX 96
CameraSizeY 96
BadColumns 10 11
BadRows 40
BadPixels 5 7 60 61 80 20
PartialBadColumn 70 1 10 50
EOF
# Header variants patched from IMOD-written files:
#  tst.mrc: ts1.mrc with titles naming a gain reference, a defect file, the
#    rotation/flip and an earlier scaling (for -titles)
#  fei.mrc: ts2.mrc without the IMOD stamp, labels or min/max, and mx = my = mz
#    = 1, the signature of a Thermo/FEI frame file (for -rfsum -1)
python3 - <<'PYEOF'
import struct
def label(text):
    b = text.encode()
    return b + b' ' * (80 - len(b))
d = bytearray(open('ts1.mrc', 'rb').read())
labs = [label('SerialEM: frames saved, r/f 1 need 0, scaled by 2.00'),
        label('   gainref.mrc   '), label('defects.txt')]
struct.pack_into('<i', d, 220, 3)
for i, l in enumerate(labs):
    d[224 + 80 * i:224 + 80 * (i + 1)] = l
open('tst.mrc', 'wb').write(d)
d = bytearray(open('ts2.mrc', 'rb').read())
struct.pack_into('<3i', d, 28, 1, 1, 1)
struct.pack_into('<2f', d, 76, 0., 0.)
struct.pack_into('<i', d, 152, 0)
struct.pack_into('<i', d, 220, 0)
d[224:1024] = b'\0' * 800
open('fei.mrc', 'wb').write(d)

# mdoc variants of ts.mdoc
base = open('ts.mdoc').read()
sects = base[base.index('[ZValue = 0]'):]
# FEI-style title with the axis to be recovered from RotationAngle
fei = ('PixelSpacing = 2.4\nDataMode = 1\nImageSize = 64 64\n\n'
       '[T = Tomography 5: TiltAxisAngle = -175.60, Defocus = -3.0]\n\n'
       '[T = Second title line]\n\n' +
       sects.replace('TiltAngle = 0.01\n', 'TiltAngle = 0.01\nRotationAngle = 85.6\n'))
open('fei.mdoc', 'w').write(fei)
# A frame-stack mdoc: global title, rotation angle in a FrameSet section
fs = ('PixelSpacing = 2.4\nDataMode = 1\nImageSize = 64 64\n'
      'T = SerialEM: frame stack of a tilt series\n\n' + sects +
      '\n[FrameSet = 0]\nRotationAngle = 175.0\n')
open('fs.mdoc', 'w').write(fs)
# Native-equivalent inputs for two titles cases whose upstream bug is fixed in
# the translation (BUGS.md, alignframes): the same titles reach the output
# without the skipped FEI title or the global T entry
open('feieq.mdoc', 'w').write(fei.replace(
    '[T = Tomography 5: TiltAxisAngle = -175.60, Defocus = -3.0]\n\n', ''))
open('fseq.mdoc', 'w').write(fs.replace(
    'T = SerialEM: frame stack of a tilt series\n\n',
    '\n[T = SerialEM: frame stack of a tilt series]\n\n'))
# Sections at binning 2 relative to the frames (for -adjust)
b2 = base.replace('ImageSize = 64 64', 'ImageSize = 32 32').replace(
    'PixelSpacing = 2.4', 'PixelSpacing = 4.8').replace(
    'binning = 1', 'binning = 2').replace('Binning = 1', 'Binning = 2')
open('bin2.mdoc', 'w').write(b2)
PYEOF
# Saved frame list: 4 sets of 8, 8, 7 and 7 frames with gaps between them, and
# tilt angles with relative frame ranges, one of whose frames were all lost
python3 -c "
nums = list(range(0, 8)) + list(range(30, 38)) + list(range(60, 67)) + list(range(90, 97))
assert len(nums) == 30
print('\\n'.join(str(n) for n in nums))" > frames.txt
printf -- '-6.0 0 7\n-3.0 30 37\n0.0 60 66\n3.0 90 96\n6.0 120 127\n' > ftilt.tlt
# (mov, movp, movs, fts, ucsf, dark, ftilt.tlt and frames.txt are still
# generated, so the seeded content of the rest is unchanged, but no longer
# kept: their cases were deleted on 2026-10-06 to keep fixtures small.)
for f in movb.mrc gainref.mrc ts1.mrc ts2.mrc ts3.mrc \
    one1.mrc one2.mrc one3.mrc one4.mrc one5.mrc one6.mrc tsstack.mrc tsstackx.mrc mov9.mrc mov3x.mrc mov9b.mrc mov3s.mrc \
    ts.mdoc fei.mdoc fs.mdoc feieq.mdoc fseq.mdoc bin2.mdoc tst.mrc fei.mrc ts.tlt dose1.txt dose2.txt dose3.txt dose5.txt list.txt defects.txt; do
  cp $f "$F"/
done
rm -rf "$S"
