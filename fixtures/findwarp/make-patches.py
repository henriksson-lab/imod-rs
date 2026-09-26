#!/usr/bin/env python3
"""Synthetic corrsearch3d-style patch files for findwarp/refinematch.

gen.py name nx ny nz nxp nyp nzp mode seed [noise] [outfrac] [missfrac] [extra]
mode: affine | warp | ident | shift
Positions on a regular grid (as corrsearch3d), vectors = x2 - x1 with
x2 = A (x1 - c) + c + d (+ smooth warp), plus gaussian noise and outliers.
extra: 0 none, 1 CCC column (id 1), 3 CCC+frac+struct (ids 1 5 6).
"""
import sys, numpy as np
a = sys.argv
name = a[1]; nx, ny, nz = map(int, a[2:5]); nxp, nyp, nzp = map(int, a[5:8])
mode = a[8]; seed = int(a[9])
noise = float(a[10]) if len(a) > 10 else 0.3
outfrac = float(a[11]) if len(a) > 11 else 0.03
missfrac = float(a[12]) if len(a) > 12 else 0.0
extra = int(a[13]) if len(a) > 13 else 1
IRREG = len(a) > 14 and a[14] == 'irreg'
rng = np.random.default_rng(seed)
cen = np.array([nx, ny, nz]) / 2.
def grid(n, np_):
    if np_ == 1: return [n // 2]
    lo, hi = int(n * 0.12), int(n * 0.88)
    if IRREG:
        return [int(round(lo + i * (hi - lo) / (np_ - 1))) for i in range(np_)]
    step = (hi - lo) // (np_ - 1)
    return [lo + i * step for i in range(np_)]
xs, ys, zs = grid(nx, nxp), grid(ny, nyp), grid(nz, nzp)
A = np.eye(3); d = np.zeros(3)
if mode in ('affine', 'warp'):
    th = rng.normal(0, 0.02, 3)
    A = A + rng.normal(0, 0.01, (3, 3))
    d = rng.normal(0, 4, 3)
elif mode == 'shift':
    d = rng.normal(0, 5, 3)
amp = rng.normal(0, 3, 3) if mode == 'warp' else np.zeros(3)
lines = []
for z in zs:
    for y in ys:
        for x in xs:
            if rng.random() < missfrac: continue
            p = np.array([x, y, z], float)
            q = A @ (p - cen) + cen + d
            q = q + amp * np.sin(2 * np.pi * p / np.array([nx, ny, nz]) * 0.7 + np.array([0.3, 1.1, 2.0]))
            v = q - p + rng.normal(0, noise, 3)
            if rng.random() < outfrac:
                v = v + rng.normal(0, 8, 3)
            ccc = 0.2 + 0.6 * rng.random()
            lines.append((x, y, z, v, ccc, rng.random(), rng.random() * 3))
with open(name, 'w') as f:
    if extra == 0:
        f.write('%7d positions\n' % len(lines))
    elif extra == 1:
        f.write('%7d positions%3d\n' % (len(lines), 1))
    else:
        f.write('%7d positions%3d%3d%3d\n' % (len(lines), 1, 5, 6))
    for (x, y, z, v, c, fr, st) in lines:
        s = '%6d%6d%6d%9.2f%9.2f%9.2f' % (x, y, z, v[0], v[1], v[2])
        if extra == 1: s += '%10.4f' % c
        elif extra == 3: s += '%10.4f%8.4f%10.5f' % (c, fr, st)
        f.write(s + '\n')
