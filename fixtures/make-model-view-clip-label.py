#!/usr/bin/env python3
"""Author an IMOD v1.2 binary model exercising the chunks no vendored fixture
carries: object labels (OLBL), contour labels (LABL), an object clip-plane
chunk (CLIP), and a VIEW chunk with object views and a model clip chunk (MCLP).

Written from IMOD/libimod/imodel_files.c and IMOD/libimod/iview.c, not from
either implementation's writer, so it is an independent check on both.
"""
import math
import struct, sys

def I(v): return struct.pack('>i', v)
def U(v): return struct.pack('>I', v)
def F(v): return struct.pack('>f', v)
def B(*v): return bytes(v)
def S(s, n): return s.encode()[:n].ljust(n, b'\0')
def P(x, y, z): return F(x) + F(y) + F(z)

CLIPSIZE = 6

def clipset(count, flags, trans, plane, normals, points):
    """Returns (header4, normals, points) for a set, padded to CLIPSIZE."""
    n = list(normals) + [(0., 0., -1.)] * (CLIPSIZE - len(normals))
    p = list(points) + [(0., 0., 0.)] * (CLIPSIZE - len(points))
    return B(count, flags, trans, plane), n, p

# `--closed` writes the same model with a closed-contour object whose contours
# are polygons: `imodinfo -e` and `-r` return immediately for an open or
# scattered object, so the ellipse fit and the length/area ratio need one.
CLOSED = '--closed' in sys.argv
if CLOSED:
    sys.argv.remove('--closed')

out = bytearray()
out += b'IMOD' + b'V1.2'
out += S('view-fixture', 128)
# xmax ymax zmax objsize flags drawmode mousemode blacklevel whitelevel
# flags: IMODF_MAT1_IS_BYTES(1<<13) | IMODF_MULTIPLE_CLIP(1<<12) | IMODF_HAS_MESH_THICK(1<<9)
flags = (1 << 13) | (1 << 12) | (1 << 9)
out += b''.join(I(v) for v in [64, 48, 20, 1, flags, 1, 1, 0, 255])
out += b''.join(F(v) for v in [0., 0., 0., 1., 1., 1.])       # offsets, scales
out += b''.join(I(v) for v in [0, 0, 0, 3, 128])              # cindex, res, thresh
out += F(1.5) + I(0) + I(0)                                   # pixsize units csum
out += b''.join(F(v) for v in [0., 0., 0.])                   # alpha beta gamma

# ---- object ----
out += b'OBJT'
out += S('clipped object', 64)
out += b'\0' * 64                                             # extra[16] ints
out += I(2)                                                   # contsize
out += U(0 if CLOSED else (1 << 9))                           # flags: closed or scattered
out += I(0) + I(1)                                            # axis drawmode
out += F(0.25) + F(0.5) + F(0.75)                             # red green blue
out += I(7)                                                   # pdrawsize
out += B(1, 3, 2, 1, 0, 0, 0)                                 # symbol..sympad
out += B(4)                                                   # trans
out += I(1)                                                   # meshsize
out += I(2)                                                   # surfsize

# OLBL: object label with two items, on surfaces 1 and 2
def label_chunk(tag, name, items):
    def padlen(s):
        # `getpadlen` (ilabel.c) rounds strlen+1 up to a multiple of 4.
        n = len(s) + 1
        return ((n // 4) + (1 if n % 4 else 0)) * 4
    body = bytearray()
    lpad = padlen(name) or 4
    body += I(len(items)) + I(lpad)
    body += name.encode().ljust(lpad, b'\0')
    for index, text in items:
        lp = padlen(text)
        body += I(index) + I(lp) + text.encode().ljust(lp, b'\0')
    size = 8 + (padlen(name) or 4) + sum(8 + padlen(t) for _, t in items)
    assert size == len(body), (size, len(body))
    return tag + I(size) + bytes(body)

out += label_chunk(b'OLBL', 'surfaces', [(1, 'top'), (2, 'bottom side')])

# ---- contours ----
if CLOSED:
    contours = [
        (1, [(20. + 12. * math.cos(t * math.pi / 6),
              16. + 5. * math.sin(t * math.pi / 6), 0.) for t in range(12)]),
        (2, [(30. + 6. * math.cos(t * math.pi / 5),
              20. + 9. * math.sin(t * math.pi / 5), 1.) for t in range(10)]),
    ]
else:
    contours = [(1, [(1., 2., 0.), (5., 6., 0.), (9., 3., 0.)]),
                (2, [(2., 4., 1.), (8., 7., 1.)])]
for co, (surf, pts) in enumerate(contours):
    out += b'CONT'
    out += I(len(pts)) + U(0) + I(0) + I(surf)
    for x, y, z in pts:
        out += P(x, y, z)
    out += label_chunk(b'LABL', 'c%d' % co, [(0, 'first point label')])

# ---- one mesh: a closed tetrahedron, so imodinfo has a surface and a volume ----
# `imodel_write_mesh` (imodel_files.c:480) writes vsize, lsize and flag as
# ints, time and surf as shorts, then vsize*3 floats and lsize ints.
# IMOD_MESH_BGNPOLYNORM (-23) makes the list normal/vertex index pairs
# (imesh.c:312).
tet = [(0., 0., 0.), (6., 0., 0.), (0., 6., 0.), (0., 0., 6.)]
faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]

def cross(u, v):
    return (u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])

verts = []          # normal, vertex, normal, vertex, ...
mlist = [-23]
for a, b, c in faces:
    u = tuple(tet[b][k] - tet[a][k] for k in range(3))
    w = tuple(tet[c][k] - tet[a][k] for k in range(3))
    n = cross(u, w)
    for idx in (a, b, c):
        mlist += [len(verts), len(verts) + 1]
        verts += [n, tet[idx]]
mlist += [-22, -1]

out += b'MESH'
out += I(len(verts)) + I(len(mlist)) + I(0)                   # vsize lsize flag
out += struct.pack('>hh', 0, 1)                               # time surf
for x, y, z in verts:
    out += P(x, y, z)
for v in mlist:
    out += I(v)

# ---- object clip planes: 2 of them ----
head, norms, points = clipset(2, 0b11, 0, 1,
                              [(0., 0., -1.), (1., 0., 0.)],
                              [(-1., -2., -3.), (-4., -5., -6.)])
out += b'CLIP' + I(SIZE_CLIP_BASE := 4 + 24 * 2)
out += head
for i in range(2):
    out += P(*norms[i])
for i in range(2):
    out += P(*points[i])

# ---- IMAT ----
out += b'IMAT' + I(16)
out += B(102, 255, 127, 4) + B(10, 20, 30, 3) + I(0) + B(0, 255, 0, 0)

# ---- VIEW chunks ----
out += b'VIEW' + I(4) + I(1)                                  # cview

# one real view
objviews = bytearray()
ovhead, ovn, ovp = clipset(3, 0b101, 6, 2,
                           [(0., 0., -1.), (0., 1., 0.), (0., 0., 1.)],
                           [(-7., -8., -9.), (-10., -11., -12.), (-13., -14., -15.)])
objviews += U(1 << 9)                                         # flags
objviews += F(0.25) + F(0.5) + F(0.75)                        # red green blue
objviews += I(7)                                              # pdrawsize
objviews += B(2, 1, 4)                                        # linewidth linesty trans
objviews += ovhead                                            # clipOut, flags, trans, plane
objviews += P(*ovn[0]) + P(*ovp[0])
objviews += B(102, 255, 127, 4)                               # ambient..shininess
objviews += B(10, 20, 30, 3)                                  # fillred..quality
objviews += I(0)                                              # mat2
objviews += B(0, 255, 0, 0)                                   # valblack..meshThickness
for i in range(1, CLIPSIZE):
    objviews += P(*ovn[i])
for i in range(1, CLIPSIZE):
    objviews += P(*ovp[i])
assert len(objviews) == 43 + 24 * CLIPSIZE, len(objviews)

mhead, mn, mp = clipset(2, 0b11, 1, 0,
                        [(0., 0., -1.), (0., -1., 0.)],
                        [(-16., -17., -18.), (-19., -20., -21.)])
mclp = b'MCLP' + I(4 + 24 * 2) + mhead
for i in range(2):
    mclp += P(*mn[i])
for i in range(2):
    mclp += P(*mp[i])

view = bytearray()
view += b''.join(F(v) for v in [45., 10., 1., 0., 1.])        # fovy rad aspect near far
view += P(0., 0., 0.)                                         # rot
view += P(0., 0., 0.)                                         # trans
view += P(1., 1., 1.)                                         # scale
view += b''.join(F(v) for v in ([1., 0., 0., 0.,
                                 0., 1., 0., 0.,
                                 0., 0., 1., 0.,
                                 0., 0., 0., 1.]))            # mat[16]
view += U(1)                                                  # world
view += S('fixture view', 32)
view += b''.join(F(v) for v in [0.2, 0.8, 0.5, 0.5, 5.])      # dcstart dcend lightx lighty plax
view += I(1) + I(len(objviews))                               # objvsize, nbwrite
view += objviews

size = 176 + 8 + len(objviews)
out += b'VIEW' + I(size) + bytes(view) + mclp

out += b'IEOF'
open(sys.argv[1], 'wb').write(bytes(out))
print("wrote", sys.argv[1], len(out), "bytes")
