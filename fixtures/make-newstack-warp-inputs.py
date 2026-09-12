#!/usr/bin/env python3
"""Author the distortion field and mag-gradient files for `tests/newstack_warp.rs`.

The formats are the ones `readDistortions` (`readdistortions.f:32`) and
`readMagGradients` (`readdistortions.f:198`) parse, written from an analytic
pattern with no RNG so they are exactly reproducible.  The grid deliberately
does not divide the 64x48 image evenly, so `getSizeAdjustedGrid` has to expand
and extrapolate it rather than use it as it stands.
"""
import math
import struct
import sys

NXG, NYG = 5, 4

lines = ["1", "64 48 1 10.0", "0 17.0 %d 0 14.0 %d" % (NXG, NYG)]
for j in range(NYG):
    row = []
    for i in range(NXG):
        dx = 0.8 * math.sin((i + 1) / 2.0) + 0.2 * j
        dy = 0.6 * math.cos((j + 1) / 1.7) - 0.15 * i
        row.append("%.4f %.4f" % (dx, dy))
    lines.append(" ".join(row))
with open(sys.argv[1] + ".idf", "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", sys.argv[1] + ".idf")

# A larger float image than the 21x13 one the other newstack fixtures use:
# the mag-gradient shift is computed from double-precision trig, and only an
# image with enough pixels shows the difference from a single-precision one.
NX, NY, NZ = 64, 48, 3
data = []
for iz in range(NZ):
    for iy in range(NY):
        for ix in range(NX):
            data.append(100.0
                        + 40.0 * math.sin(ix / 5.0)
                        + 25.0 * math.cos(iy / 3.5)
                        + 7.0 * iz
                        + 3.0 * ((ix * 7 + iy * 13 + iz * 29) % 11))
amin, amax = min(data), max(data)
amean = sum(data) / len(data)
h = bytearray(1024)
struct.pack_into('<3i', h, 0, NX, NY, NZ)
struct.pack_into('<i', h, 12, 2)
struct.pack_into('<3i', h, 16, 0, 0, 0)
struct.pack_into('<3i', h, 28, NX, NY, NZ)
struct.pack_into('<3f', h, 40, float(NX), float(NY), float(NZ))
struct.pack_into('<3f', h, 52, 90.0, 90.0, 90.0)
struct.pack_into('<3i', h, 64, 1, 2, 3)
struct.pack_into('<3f', h, 76, amin, amax, amean)
struct.pack_into('<3f', h, 196, 0.0, 0.0, 0.0)
h[208:212] = b'MAP '
h[212:216] = bytes([68, 65, 0, 0])
struct.pack_into('<f', h, 216, 0.0)
struct.pack_into('<i', h, 220, 1)
label = b'newstack warping fixture'
h[224:224 + len(label)] = label
with open(sys.argv[1] + "-input.mrc", "wb") as f:
    f.write(bytes(h))
    f.write(struct.pack('<%df' % len(data), *data))
print("wrote", sys.argv[1] + "-input.mrc")

grad = ["1", "5 12.5 5.0"]
for i in range(5):
    tilt = -20.0 + 10.0 * i
    grad.append("%.1f %.3f %.4f" % (tilt, 0.35 - 0.017 * i, 0.12 - 0.0075 * i))
with open(sys.argv[1] + ".mgt", "w") as f:
    f.write("\n".join(grad) + "\n")
print("wrote", sys.argv[1] + ".mgt")
