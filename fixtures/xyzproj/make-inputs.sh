#!/bin/bash
# Regenerates the seeded xyzproj inputs in this directory with the native
# raw2mrc (so IMOD itself writes the MRC container), and the tilt-angle files.
set -e
cd "$(dirname "$0")"
R=${IMOD_REF:-/tmp/imod-reference-build}
python3 - <<'PY'
import numpy as np
rng=np.random.default_rng(11)
def mk(name,nx,ny,nz,dt,scale=1.0,off=0.0):
    z,y,x=np.mgrid[0:nz,0:ny,0:nx]
    a=(np.sin(x/3.0)+np.cos(y/4.0+z/5.0)+0.05*x*z/8)*40+rng.normal(0,4,(nz,ny,nx))+off
    (a*scale).astype(dt).tofile(name)
mk("v.raw",22,18,14,np.float32)
mk("s.raw",15,13,9,np.int16,scale=7)
mk("b.raw",15,13,9,np.uint8,off=100)
mk("ts.raw",36,30,9,np.float32)
with open("ts.tlt","w") as f:
    for a in range(-48,49,12): f.write("%7.2f\n"%a)
with open("v.tlt","w") as f:
    for a in (-50,-20,0,5.5,33,70): f.write("%7.2f\n"%a)
PY
export LD_LIBRARY_PATH=$R/buildlib
$R/mrc/raw2mrc -x 22 -y 18 -z 14 -t float v.raw v.mrc >/dev/null
$R/mrc/raw2mrc -x 15 -y 13 -z 9 -t short s.raw s.mrc >/dev/null
$R/mrc/raw2mrc -x 15 -y 13 -z 9 -t byte b.raw b.mrc >/dev/null
$R/mrc/raw2mrc -x 36 -y 30 -z 9 -t float ts.raw ts.mrc >/dev/null
rm -f *.raw
