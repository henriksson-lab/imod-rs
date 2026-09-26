#!/bin/bash
# Regenerates the seeded assemblevol inputs in this directory with the native
# raw2mrc (so IMOD itself writes the MRC container).
set -e
cd "$(dirname "$0")"
R=${IMOD_REF:-/tmp/imod-reference-build}
python3 - <<'PY'
import numpy as np
rng=np.random.default_rng(3)
def mk(name,nx,ny,nz,dt,off=0,scale=1.0):
    z,y,x=np.mgrid[0:nz,0:ny,0:nx]
    a=(np.sin(x/3.0+off)+np.cos(y/4.0)+0.3*z)*50+rng.normal(0,3,(nz,ny,nx))
    (a*scale).astype(dt).tofile(name)
k=0
for iz in range(2):
  for iy in range(2):
    for ix in range(3):
      mk(f"p{ix}{iy}{iz}.raw",13,11,5,np.float32,off=k); k+=1
mk("s0.raw",9,7,4,np.int16,scale=5); mk("s1.raw",9,7,4,np.int16,off=1,scale=5)
mk("b0.raw",9,7,4,np.uint8); mk("b1.raw",9,7,4,np.uint8,off=2)
PY
export LD_LIBRARY_PATH=$R/buildlib
for f in p*.raw; do $R/mrc/raw2mrc -x 13 -y 11 -z 5 -t float $f ${f%.raw}.mrc >/dev/null; done
for f in s0 s1; do $R/mrc/raw2mrc -x 9 -y 7 -z 4 -t short $f.raw $f.mrc >/dev/null; done
for f in b0 b1; do $R/mrc/raw2mrc -x 9 -y 7 -z 4 -t byte $f.raw $f.mrc >/dev/null; done
rm -f *.raw
