#!/usr/bin/env python3
"""Author the inputs of `tests/blendmont_cli.rs` in fixtures/blendmont.

bm.st is a seeded synthetic 3 x 3 montage of 64 x 56 pieces with overlaps of
20 in X and Y (so every edge function grid has at least two rows), 2 sections,
small random piece displacements and per-piece intensity scaling, cut from a
smoothed-noise image with blobs; written as shorts by the native `raw2mrc`.
bm.pl is its piece list.  bmn.pl is the same list with negative numbers (a
multinegative section).  g.txt is a version 1 mag gradient file, x.xf two g
transforms, excl.txt the points of the exclusion model excl.mod (made with the
native `point2model -image bm.st`).  Needs numpy and the reference build
(REF, default /tmp/imod-reference-build).
"""
import os, subprocess
import numpy as np
REF = os.environ.get('REF', '/tmp/imod-reference-build')
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blendmont')
os.makedirs(here, exist_ok=True)
os.chdir(here)
env = dict(os.environ, LD_LIBRARY_PATH=REF + '/buildlib')

def smooth(rng, a, k):
    from numpy.fft import rfft2, irfft2
    ky = np.fft.fftfreq(a.shape[0])[:, None]
    kx = np.fft.rfftfreq(a.shape[1])[None, :]
    return irfft2(rfft2(a) * np.exp(-(kx**2 + ky**2) * (2 * np.pi * k)**2 / 2), s=a.shape)

rng = np.random.default_rng(2026)
nx, ny, nxp, nyp, ovx, ovy, nsec, shift = 64, 56, 3, 3, 20, 20, 2, 3
sx, sy = nx - ovx, ny - ovy
W, H = (nxp - 1) * sx + nx + 4 * shift + 20, (nyp - 1) * sy + ny + 4 * shift + 20
pieces, pl = [], []
for z in range(nsec):
    base = smooth(rng, rng.normal(0, 1, (H, W)), 2.0) * 60 + \
        smooth(rng, rng.normal(0, 1, (H, W)), 6.0) * 120 + 500
    yy, xx = np.mgrid[0:H, 0:W]
    for b in range(int(W * H / 1500)):
        cx, cy, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(2, 6)
        base += 150 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * r * r))
    for iy in range(nyp):
        for ix in range(nxp):
            dx, dy = int(rng.integers(-shift, shift + 1)), int(rng.integers(-shift, shift + 1))
            x0, y0 = 10 + 2 * shift + ix * sx + dx, 10 + 2 * shift + iy * sy + dy
            p = base[y0:y0 + ny, x0:x0 + nx] + rng.normal(0, 4, (ny, nx))
            p = p * (1 + 0.08 * rng.uniform(-1, 1))
            pieces.append(p.astype(np.float32))
            pl.append((ix * sx, iy * sy, z))
np.clip(np.array(pieces), -32000, 32000).astype('<i2').tofile('bm.raw')
subprocess.run([REF + '/mrc/raw2mrc', '-x', str(nx), '-y', str(ny), '-z', str(len(pieces)),
                '-t', 'short', 'bm.raw', 'bm.st'], check=True, env=env,
               stdout=subprocess.DEVNULL)
os.remove('bm.raw')
with open('bm.pl', 'w') as f:
    for x, y, z in pl:
        f.write('%d %d %d\n' % (x, y, z))
with open('bmn.pl', 'w') as f:
    for x, y, z in pl:
        f.write('%d %d %d %d\n' % (x, y, z, 1 if x < sx else 2))
with open('g.txt', 'w') as f:
    f.write('1\n2 10.0 -3.0\n0.0 0.5 0.2\n30.0 -0.4 0.3\n')
with open('x.xf', 'w') as f:
    f.write('0.9998 -0.0175 0.0175 0.9998 3.5 -2.25\n1.01 0.02 -0.015 0.99 -4.0 6.5\n')
with open('excl.txt', 'w') as f:
    f.write('1 1 54 30 0\n1 1 98 110 1\n2 1 30 46 0\n')
subprocess.run([REF + '/imodutil/point2model', '-image', 'bm.st', 'excl.txt', 'excl.mod'],
               check=True, env=env, stdout=subprocess.DEVNULL)
