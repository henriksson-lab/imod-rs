#!/usr/bin/env python3
"""Author the MRC files that `tests/newstack_mixed_inputs.rs` stacks together.

Written straight to the MRC layout in `IMOD/libiimod/mrcfiles.h` so the inputs
are independent of either implementation's writer.  The three files differ in
mode and in X/Y size, which is what makes `newstack` recompute `rescale` and
`getReducedSize` for every input file rather than once from the first.
"""
import math
import struct
import sys

MODE_PACK = {0: ('<%db', 1), 1: ('<%dh', 2), 2: ('<%df', 4)}


def pattern(nx, ny, nz, seed):
    values = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                values.append(0.5
                              + 0.35 * math.sin((ix + seed) / 3.0)
                              + 0.2 * math.cos(iy / 2.5)
                              + 0.03 * iz
                              + 0.01 * ((ix * 7 + iy * 13 + iz * 29 + seed) % 11))
    return values


def write(path, nx, ny, nz, mode, seed, scale, offset, label):
    values = pattern(nx, ny, nz, seed)
    data = [v * scale + offset if mode == 2
            else int(round(v * scale)) + offset for v in values]
    amin, amax = float(min(data)), float(max(data))
    amean = float(sum(data)) / len(data)
    h = bytearray(1024)
    struct.pack_into('<3i', h, 0, nx, ny, nz)
    struct.pack_into('<i', h, 12, mode)
    struct.pack_into('<3i', h, 16, 0, 0, 0)
    struct.pack_into('<3i', h, 28, nx, ny, nz)
    struct.pack_into('<3f', h, 40, float(nx), float(ny), float(nz))
    struct.pack_into('<3f', h, 52, 90.0, 90.0, 90.0)
    struct.pack_into('<3i', h, 64, 1, 2, 3)
    struct.pack_into('<3f', h, 76, amin, amax, amean)
    struct.pack_into('<3f', h, 196, 0.0, 0.0, 0.0)
    h[208:212] = b'MAP '
    h[212:216] = bytes([68, 65, 0, 0])
    struct.pack_into('<f', h, 216, 0.0)
    struct.pack_into('<i', h, 220, 1)
    h[224:224 + len(label)] = label
    fmt, width = MODE_PACK[mode]
    with open(path, 'wb') as f:
        f.write(bytes(h))
        f.write(struct.pack(fmt % len(data), *data))
    print("wrote", path, 1024 + width * len(data), "bytes")


# Mode 0 is signed in the file, so the byte pattern is centred on zero.
write(sys.argv[1] + '-byte.mrc', 21, 13, 3, 0, 0, 150.0, -80, b'newstack mixed byte fixture')
write(sys.argv[1] + '-short.mrc', 21, 13, 3, 1, 5, 20000.0, 0, b'newstack mixed short fixture')
write(sys.argv[1] + '-small.mrc', 11, 7, 3, 0, 9, 150.0, -80, b'newstack mixed small fixture')
