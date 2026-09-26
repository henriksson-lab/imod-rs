#!/bin/bash
# Regenerates the seeded xcorrstack inputs in this directory with the native
# raw2mrc (so IMOD itself writes the MRC container).
set -e
cd "$(dirname "$0")"
R=${IMOD_REF:-/tmp/imod-reference-build}
python3 - <<'PY'
import numpy as np
rng=np.random.default_rng(7)
def mk(name,nx,ny,nz,dt,scale=1.0):
    y,x=np.mgrid[0:ny,0:nx]
    arr=[]
    for z in range(nz):
        img=np.exp(-((x-nx/2-2*z)**2+(y-ny/2+z)**2)/(2*(nx/8)**2))*100+rng.normal(0,5,(ny,nx))
        arr.append(img)
    (np.array(arr)*scale).astype(dt).tofile(name)
mk("st.raw",30,24,3,np.float32); mk("single.raw",30,24,1,np.float32); mk("small.raw",18,14,1,np.float32)
mk("stb.raw",32,20,2,np.uint8); mk("singleb.raw",32,20,1,np.uint8)
mk("sts.raw",25,19,2,np.int16,10.0); mk("singles_small.raw",12,19,1,np.int16,10.0)
mk("big.raw",40,24,1,np.float32)
PY
export LD_LIBRARY_PATH=$R/buildlib
M=$R/mrc/raw2mrc
$M -x 30 -y 24 -z 3 -t float st.raw st.mrc; $M -x 30 -y 24 -z 1 -t float single.raw single.mrc
$M -x 18 -y 14 -z 1 -t float small.raw small.mrc; $M -x 32 -y 20 -z 2 -t byte stb.raw stb.mrc
$M -x 32 -y 20 -z 1 -t byte singleb.raw singleb.mrc; $M -x 25 -y 19 -z 2 -t short sts.raw sts.mrc
$M -x 12 -y 19 -z 1 -t short singles_small.raw singles_small.mrc; $M -x 40 -y 24 -z 1 -t float big.raw big.mrc
rm -f *.raw
