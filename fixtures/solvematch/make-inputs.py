#!/usr/bin/env python3
"""Synthetic dual-axis fiducial sets for solvematch differentials.

Truth: bead positions P in A-tomogram centered index coords (x, depth, slice).
B tomogram coords Q satisfy P = M Q + d, M = rotation about the depth axis by
~90 deg (plus small extras / scale), so B is the 90-degree rotated axis.
Fiducial .xyz files are in tiltalign point-file format: aligned-stack x, y
(0..Dim) and z (depth, sign flipped relative to tomogram as solvematch expects).
"""
import numpy as np, os, sys, math

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
out = sys.argv[2] if len(sys.argv) > 2 else 'data'
os.makedirs(out, exist_ok=True)

def rotz(deg):  # rotation in the x-slice plane (about depth axis)
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    # coordinate order (x, depth, slice)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

def rotx(deg):  # small tilt about x (mixes depth and slice)
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def write_xyz(path, pts, ids, objs, conts, pix, dim, nocols=False, header=True, oldpix=False):
    # pts are aligned-stack (x, y, zfid) with x,y already offset by Dim/2
    with open(path, 'w') as f:
        for k, (p, i, o, c) in enumerate(zip(pts, ids, objs, conts)):
            if nocols:
                line = "%4d %9.2f %9.2f %9.2f" % (i, p[0], p[1], p[2])
            else:
                line = "%4d %9.2f %9.2f %9.2f %6d %4d" % (i, p[0], p[1], p[2], o, c)
            if k == 0 and header:
                if oldpix:
                    line += " Pixel size: %11.5f" % pix
                else:
                    line += " Pix: %11.5f Dim: %5d %5d" % (pix, dim[0], dim[1])
            f.write(line + "\n")

def make(name, nbead=30, surfaces=2, thick=60, size=400, rot=91.5, tiltx=1.0,
         shift=(3.0, -2.0, 4.0), mag=1.0, noise=0.25, permute=True,
         invert=False, pixA=1.0, pixB=1.0, outlier=False, nocols=False, xtiltA=0.0, xtiltB=0.0,
         distort=0.0, bigids=False):
    half = size / 2 - 20
    P = np.zeros((nbead, 3))
    P[:, 0] = rng.uniform(-half, half, nbead)
    P[:, 2] = rng.uniform(-half, half, nbead)
    if surfaces == 2:
        P[:, 1] = np.where(np.arange(nbead) % 2 == 0, thick / 2, -thick / 2) + rng.normal(0, 1.0, nbead)
    else:
        P[:, 1] = thick / 3 + rng.normal(0, 0.3, nbead)
    M = rotz(rot) @ rotx(tiltx) * mag
    if invert:
        M = M @ np.diag([1, -1, 1])
    d = np.array(shift)
    Q = (np.linalg.inv(M) @ (P - d).T).T
    PA = P + rng.normal(0, noise, P.shape)
    QB = Q + rng.normal(0, noise, Q.shape)
    if outlier:
        QB[5] += np.array([15.0, 3.0, -12.0])
    if distort:
        QB[:, 0] += distort * np.sin(math.pi * Q[:, 2] / size)
        QB[:, 2] += distort * np.cos(math.pi * Q[:, 0] / size) + distort
    dim = (size, size)
    # A aligned coords: x = X*scaleA^-1 ... tomogram delta == fid pixel -> scale 1
    def to_fid(T, pix_fid, delta):
        s = delta / pix_fid  # fid pixels per tomogram pixel
        ali = np.zeros_like(T)
        ali[:, 0] = T[:, 0] * s + dim[0] / 2
        ali[:, 1] = T[:, 2] * s + dim[1] / 2
        ali[:, 2] = -T[:, 1] * s
        return ali
    fa = to_fid(PA, pixA, 1.0)
    fb = to_fid(QB, pixB, 1.0)
    idsA = np.arange(1, nbead + 1)
    if bigids:
        idsA = idsA + 10000
    perm = rng.permutation(nbead) if permute else np.arange(nbead)
    idsB = np.empty(nbead, int)
    idsB[perm] = np.arange(1, nbead + 1)   # bead k is point idsB[k] in B
    order = np.argsort(idsB)
    objA = np.ones(nbead, int); contA = idsA
    objB = np.ones(nbead, int); contB = idsB
    write_xyz(f"{out}/{name}A.xyz", fa, idsA, objA, contA, pixA, dim, nocols)
    write_xyz(f"{out}/{name}B.xyz", fb[order], idsB[order], objB[order], contB[order], pixB, dim, nocols)
    # correspondence list: B point number for A points 1..nbead
    with open(f"{out}/{name}.corr", 'w') as f:
        f.write(",".join(str(x) for x in idsB) + "\n")
    # matching models: tomogram index coordinates (x, depth, slice)
    ntom = (size, thick + 40, size)
    for tag, T in (("A", PA), ("B", QB)):
        with open(f"{out}/{name}{tag}match.txt", 'w') as f:
            for k in range(0, nbead, 2):
                x = T[k, 0] + ntom[0] / 2; y = T[k, 1] + ntom[1] / 2; z = T[k, 2] + ntom[2] / 2
                f.write("1 %d %.3f %.3f %.3f\n" % (k // 2 + 1, x, y, z))
    # 2D fiducial models (projection at zero tilt, section izBest) and transfer file
    izA, izB = 20, 22
    for tag, F, ids, iz in (("A", fa, idsA, izA), ("B", fb, idsB, izB)):
        with open(f"{out}/{name}{tag}fid.txt", 'w') as f:
            for k in range(nbead):
                c = ids[k]
                for z in range(iz - 2, iz + 3):
                    f.write("1 %d %.3f %.3f %d\n" % (c, F[k, 0] + 0.1 * (z - iz), F[k, 1], z))
    with open(f"{out}/{name}.transfer", 'w') as f:
        f.write("%d %d 0\n" % (izA, izB))
        for k in range(0, nbead, 1):
            if k % 7 == 3:
                continue
            f.write("%.2f %.2f %.2f %.2f\n" % (fa[k, 0], fa[k, 1], fb[k, 0], fb[k, 1]))
    with open(f"{out}/{name}.info", 'w') as f:
        f.write("M=%s\nd=%s\ntom=%s\n" % (M.tolist(), d.tolist(), ntom))

make("two", surfaces=2)
make("one", surfaces=1, nbead=24)
make("inv", surfaces=1, nbead=24, invert=True)
make("out", surfaces=2, outlier=True, nbead=40)
make("scl", surfaces=2, pixA=0.5, pixB=0.5)
make("nocol", surfaces=2, nocols=True, permute=False)
make("aniso", surfaces=2, mag=1.0, nbead=30, noise=0.2)
make("big", surfaces=2, nbead=200, size=1000, thick=150, noise=1.2)
make("warp", surfaces=2, nbead=150, size=800, thick=100, noise=0.1, distort=3.0)
make("hiid", surfaces=2, nbead=10, bigids=True, permute=False)
