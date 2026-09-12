#!/usr/bin/env python3
"""Author the small MRC volumes that `tests/mrcsec_sections.rs` reads.

One per storage mode that `mrcsec.c` converts between: 0 (signed byte), 1
(signed short), 2 (float), 6 (unsigned short) and 4 (complex float).  Written
straight to the MRC layout in `IMOD/libiimod/mrcfiles.h` from an analytic
pattern with no RNG, so they are reproducible anywhere.  The sizes are odd on
purpose so the sub-rectangle reads in the test are not aligned.
"""
import math
import struct
import sys
import os

NX, NY, NZ = 21, 13, 4

def sample(ix, iy, iz):
    return (120.0
            + 40.0 * math.sin(ix / 3.0)
            + 25.0 * math.cos(iy / 2.5)
            + 9.0 * iz
            + 3.0 * ((ix * 7 + iy * 13 + iz * 29) % 11))

def write(path, mode, pack, values, nx=NX):
    amin, amax = min(values), max(values)
    amean = sum(values) / len(values)
    h = bytearray(1024)
    struct.pack_into('<3i', h, 0, nx, NY, NZ)
    struct.pack_into('<i', h, 12, mode)
    struct.pack_into('<3i', h, 16, 0, 0, 0)
    struct.pack_into('<3i', h, 28, nx, NY, NZ)
    struct.pack_into('<3f', h, 40, float(nx), float(NY), float(NZ))
    struct.pack_into('<3f', h, 52, 90.0, 90.0, 90.0)
    struct.pack_into('<3i', h, 64, 1, 2, 3)
    struct.pack_into('<3f', h, 76, float(amin), float(amax), float(amean))
    struct.pack_into('<3f', h, 196, 0.0, 0.0, 0.0)
    h[208:212] = b'MAP '
    h[212:216] = bytes([68, 65, 0, 0])
    struct.pack_into('<f', h, 216, -1.0)
    struct.pack_into('<i', h, 220, 1)
    label = b'mrcsec section fixture'
    h[224:224 + len(label)] = label
    with open(path, 'wb') as f:
        f.write(bytes(h))
        f.write(pack(values))
    print("wrote", path, os.path.getsize(path), "bytes")

out = sys.argv[1]
grid = [sample(ix, iy, iz)
        for iz in range(NZ) for iy in range(NY) for ix in range(NX)]

# mode 0: signed byte, as IMOD writes it with the signed-bytes flag off
bytes0 = [max(0, min(255, int(round(v)))) for v in grid]
write(os.path.join(out, 'mrcsec-mode0.mrc'), 0,
      lambda v: struct.pack('<%dB' % len(v), *v), bytes0)

# mode 1: signed short
short1 = [max(-32768, min(32767, int(round(v * 100)))) for v in grid]
write(os.path.join(out, 'mrcsec-mode1.mrc'), 1,
      lambda v: struct.pack('<%dh' % len(v), *v), short1)

# mode 2: float
write(os.path.join(out, 'mrcsec-mode2.mrc'), 2,
      lambda v: struct.pack('<%df' % len(v), *v), grid)

# mode 6: unsigned short
ushort6 = [max(0, min(65535, int(round(v * 200)))) for v in grid]
write(os.path.join(out, 'mrcsec-mode6.mrc'), 6,
      lambda v: struct.pack('<%dH' % len(v), *v), ushort6)

# mode 4: complex float -- two floats per pixel, so the row is 2 * nx long
cx = []
for iz in range(NZ):
    for iy in range(NY):
        for ix in range(NX):
            cx.append(sample(ix, iy, iz) / 100.0)
            cx.append(sample(iy, ix, iz) / 100.0)
write(os.path.join(out, 'mrcsec-mode4.mrc'), 4,
      lambda v: struct.pack('<%df' % len(v), *v), cx)
