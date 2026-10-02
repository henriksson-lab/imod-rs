#!/usr/bin/env python3
"""Synthetic gold-bead tilt series for the RAPTOR golden suite.

make-syn1.py NAME NX NY TMIN TMAX TSTEP NBEAD SEED SIGMA
Writes NAME.mrc (bytes, through the native raw2mrc and newstack) and NAME.rawtlt.
Dark Gaussian beads at random 3D positions, tilted about Y with a 4-degree axis
rotation, per-view shifts and noise (after /big/henriksson/realbench/wave6-beadtrack/gen.py).
"""
import os, subprocess, sys
import numpy as np
REF = "/tmp/imod-reference-build"
ENV = dict(os.environ, AUTODOC_DIR=REF + "/autodoc", LD_LIBRARY_PATH=REF + "/buildlib")
name = sys.argv[1]
nx, ny = int(sys.argv[2]), int(sys.argv[3])
tmin, tmax, tstep = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
nbead, seed, sigma = int(sys.argv[7]), int(sys.argv[8]), float(sys.argv[9])
rng = np.random.default_rng(seed)
tilts = np.arange(tmin, tmax + 0.001, tstep)
nz = len(tilts)
rad = np.deg2rad(4.0)
margin = 14
bx = rng.uniform(-(nx / 2 - margin), nx / 2 - margin, nbead)
by = rng.uniform(-(ny / 2 - margin), ny / 2 - margin, nbead)
bz = rng.uniform(-30, 30, nbead)
shifts = rng.normal(0, 1.5, (nz, 2))
yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
stack = np.empty((nz, ny, nx), np.float32)
for iv, t in enumerate(tilts):
    tr = np.deg2rad(t)
    xp = bx * np.cos(tr) + bz * np.sin(tr)
    xr = np.cos(rad) * xp - np.sin(rad) * by + shifts[iv, 0] + nx / 2
    yr = np.sin(rad) * xp + np.cos(rad) * by + shifts[iv, 1] + ny / 2
    img = 150.0 + rng.normal(0, 6.0, (ny, nx))
    for x0, y0 in zip(xr, yr):
        img += -80.0 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma * sigma))
    stack[iv] = img
stack.astype(np.float32).tofile(name + ".raw")
with open(name + ".rawtlt", "w") as f:
    for t in tilts:
        f.write("%.2f\n" % t)
subprocess.run([REF + "/mrc/raw2mrc", "-x", str(nx), "-y", str(ny), "-z", str(nz), "-t", "float",
                name + ".raw", name + "_f.mrc"], env=ENV, check=True, stdout=subprocess.DEVNULL)
subprocess.run([REF + "/flib/image/newstack", "-mode", "0", "-scale", "0,255", name + "_f.mrc", name + ".mrc"],
               env=ENV, check=True, stdout=subprocess.DEVNULL)
os.remove(name + ".raw"); os.remove(name + "_f.mrc")
