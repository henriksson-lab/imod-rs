#!/usr/bin/env python3
"""Author the small MRC volume that `tests/newstack_reduce.rs` reduces.

Written straight to the MRC layout in `IMOD/libiimod/mrcfiles.h` so it is
independent of either implementation's writer.  The sizes deliberately are not
multiples of the binning factors used in the test: that is what makes
`getBinnedSize`'s X and Y offsets nonzero.
"""
import math
import struct
import sys

nx, ny, nz = 21, 13, 3
data = []
for iz in range(nz):
    for iy in range(ny):
        for ix in range(nx):
            # A fixed analytic pattern: no RNG, so this is exactly
            # reproducible on any machine.
            v = (100.0
                 + 40.0 * math.sin(ix / 3.0)
                 + 25.0 * math.cos(iy / 2.5)
                 + 7.0 * iz
                 + 3.0 * ((ix * 7 + iy * 13 + iz * 29) % 11))
            data.append(v)

amin, amax = min(data), max(data)
amean = sum(data) / len(data)

h = bytearray(1024)
struct.pack_into('<3i', h, 0, nx, ny, nz)
struct.pack_into('<i', h, 12, 2)                       # mode 2 = float
struct.pack_into('<3i', h, 16, 0, 0, 0)                # nxstart, nystart, nzstart
struct.pack_into('<3i', h, 28, nx, ny, nz)             # mx, my, mz
struct.pack_into('<3f', h, 40, float(nx), float(ny), float(nz))  # xlen, ylen, zlen
struct.pack_into('<3f', h, 52, 90.0, 90.0, 90.0)       # alpha, beta, gamma
struct.pack_into('<3i', h, 64, 1, 2, 3)                # mapc, mapr, maps
struct.pack_into('<3f', h, 76, amin, amax, amean)
struct.pack_into('<h', h, 96, 0)                       # ispg via idtype area
struct.pack_into('<3f', h, 196, 0.0, 0.0, 0.0)         # xorg, yorg, zorg
h[208:212] = b'MAP '
h[212:216] = bytes([68, 65, 0, 0])                     # little-endian stamp
struct.pack_into('<f', h, 216, 0.0)                    # rms
struct.pack_into('<i', h, 220, 1)                      # nlabl
label = b'newstack reduction fixture'
h[224:224 + len(label)] = label

with open(sys.argv[1], 'wb') as f:
    f.write(bytes(h))
    f.write(struct.pack('<%df' % len(data), *data))
print("wrote", sys.argv[1], 1024 + 4 * len(data), "bytes")
