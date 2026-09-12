#!/usr/bin/env python3
"""Author the *large* distortion, mag-gradient and warping files that force
`newstack`'s chunked route to run several chunks.

`fixtures/make-newstack-warp-inputs.py` authors a 64x48x3 volume, which stays
in a single chunk at every memory limit `newstack` accepts.  The claim this
fixture exists to test is about the grid geometry "from the second chunk on",
so it needs an image whose output does not fit in the working array: 1200x900
is 1,080,000 pixels a section, so `-memory 4` (1,048,576 elements) and the
`-test` pairs below split it two, three and more ways.

Written from an analytic pattern with no RNG, so it is exactly reproducible.
The grids deliberately do not divide the image evenly, so `getSizeAdjustedGrid`
has to expand and extrapolate them rather than use them as they stand.

Usage: make-newstack-warp-big-inputs.py <prefix> [nx ny nz]
writes <prefix>.idf, <prefix>.mgt, <prefix>.xf and <prefix>-input.mrc.  The
committed fixture files are the default 1200x900x3; a size is given only to
author the much larger scratch volume that `-memory`, whose smallest accepted
limit is 39 MB, needs before it will chunk at all.
"""
import math
import struct
import sys

NX, NY, NZ = 1200, 900, 3
if len(sys.argv) > 4:
    NX, NY, NZ = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

# --- version-1 distortion field (`readDistortions`, `readdistortions.f:32`) ---
NXG, NYG = 8, 7
lines = ["1", "%d %d 1 10.0" % (NX, NY), "0 170.0 %d 0 140.0 %d" % (NXG, NYG)]
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

# --- mag-gradient table (`readMagGradients`, `readdistortions.f:198`) ---
grad = ["1", "5 12.5 5.0"]
for i in range(5):
    tilt = -20.0 + 10.0 * i
    grad.append("%.1f %.3f %.4f" % (tilt, 0.35 - 0.017 * i, 0.12 - 0.0075 * i))
with open(sys.argv[1] + ".mgt", "w") as f:
    f.write("\n".join(grad) + "\n")
print("wrote", sys.argv[1] + ".mgt")

# --- version-3 warping file (`writeWarpFile`, `warpfiles.c:353`) ---
xf = ["3", "%d %d %d 1 12.500000 1" % (NX, NY, NZ)]
for iz in range(NZ):
    xf.append("0.000000  170.000000  %d  0.000000  140.000000  %d" % (NXG, NYG))
    xf.append("1.000000  0.000000  0.000000  1.000000  %.3f %.3f"
              % (0.5 * iz, -0.3 * iz))
    for j in range(NYG):
        out = ""
        for i in range(NXG):
            vx = 0.9 * math.sin((i + 1 + 2 * iz) / 2.0) + 0.15 * j
            vy = 0.7 * math.cos((j + 1) / 1.7) - 0.12 * i + 0.05 * iz
            out += "  %.3f  %.3f" % (vx, vy)
            if i % 4 == 3 or i == NXG - 1:
                xf.append(out)
                out = ""
with open(sys.argv[1] + ".xf", "w") as f:
    f.write("\n".join(xf) + "\n")
print("wrote", sys.argv[1] + ".xf")

# --- the float volume itself (mode 2; see the small script for why) ---
# Every value is a multiple of 0.25 in the range 60..240, so it is exact in
# both `float` and `double` and `tests/newstack_warp_chunked.rs` can author the
# identical volume from the same expression without depending on either
# runtime's libm.
h = bytearray(1024)
struct.pack_into('<3i', h, 0, NX, NY, NZ)
struct.pack_into('<i', h, 12, 2)
struct.pack_into('<3i', h, 16, 0, 0, 0)
struct.pack_into('<3i', h, 28, NX, NY, NZ)
struct.pack_into('<3f', h, 40, float(NX), float(NY), float(NZ))
struct.pack_into('<3f', h, 52, 90.0, 90.0, 90.0)
struct.pack_into('<3i', h, 64, 1, 2, 3)
struct.pack_into('<3f', h, 196, 0.0, 0.0, 0.0)
h[208:212] = b'MAP '
h[212:216] = bytes([68, 65, 0, 0])
struct.pack_into('<f', h, 216, 0.0)
struct.pack_into('<i', h, 220, 1)
label = b'newstack chunked warping fixture'
h[224:224 + len(label)] = label

amin, amax, asum = 1.0e37, -1.0e37, 0.0
sections = []
for iz in range(NZ):
    sec = []
    for iy in range(NY):
        for ix in range(NX):
            v = (100.0 + 0.25 * ((ix * 7 + iy * 13 + iz * 29) % 401)
                 + 0.5 * (ix % 53) - 0.75 * (iy % 71) + 3.0 * iz)
            sec.append(v)
    amin = min(amin, min(sec))
    amax = max(amax, max(sec))
    asum += sum(sec)
    sections.append(sec)
struct.pack_into('<3f', h, 76, amin, amax, asum / (NX * NY * NZ))
with open(sys.argv[1] + "-input.mrc", "wb") as f:
    f.write(bytes(h))
    for sec in sections:
        f.write(struct.pack('<%df' % len(sec), *sec))
print("wrote", sys.argv[1] + "-input.mrc")
