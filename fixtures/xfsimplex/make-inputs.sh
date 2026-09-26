#!/bin/bash
# Regenerates the seeded xfsimplex inputs with the native raw2mrc: a smooth
# random image pair (B is A rotated by 4 degrees, stretched 2% and shifted),
# two sections each, float and short, plus initial transform files.  A larger
# 128x104 pair (al/bl) is still drawn, but only the 60x56 odd.mrc cut from it
# is kept: the cases that ran on al/bl run on a/b since 2026-09-26.
set -e
cd "$(dirname "$0")"
R=${IMOD_REF:-/tmp/imod-reference-build}
python3 - <<'PY'
import numpy as np
from scipy import ndimage
rng=np.random.default_rng(21)
def pair(nx,ny):
    base=ndimage.gaussian_filter(rng.normal(0,1,(ny+40,nx+40)),3)*40+100
    a=base[20:20+ny,20:20+nx]
    th=np.radians(4.0); c,s=np.cos(th),np.sin(th)
    cy,cx=(ny+40)/2,(nx+40)/2
    mat=np.array([[c*1.02,-s],[s,c]])
    off=np.array([cy,cx])-mat@np.array([cy,cx])+np.array([1.7,-2.3])
    b=ndimage.affine_transform(base,mat,offset=off,order=1)[20:20+ny,20:20+nx]
    b=b+rng.normal(0,1,b.shape)
    return a,b
a1,b1=pair(64,56); a2,b2=pair(64,56)
np.stack([a1,a2]).astype(np.float32).tofile("a.raw")
np.stack([b1,b2]).astype(np.float32).tofile("b.raw")
np.stack([b1,b2]).astype(np.int16).tofile("bs.raw")
a3,b3=pair(128,104)
a3.astype(np.float32).tofile("al.raw"); b3.astype(np.float32).tofile("bl.raw")
open("init.xf","w").write("   1.0000000   0.0000000   0.0000000   1.0000000       0.000       0.000\n   0.9950000  -0.0700000   0.0700000   0.9950000       1.500      -2.000\n")
open("bad.xf","w").write("1 0 0 x 0 0\n")
PY
export LD_LIBRARY_PATH=$R/buildlib
rm -f a.mrc b.mrc bs.mrc odd.mrc  # else raw2mrc leaves `~` backups
$R/mrc/raw2mrc -x 64 -y 56 -z 2 -t float a.raw a.mrc >/dev/null
$R/mrc/raw2mrc -x 64 -y 56 -z 2 -t float b.raw b.mrc >/dev/null
$R/mrc/raw2mrc -x 64 -y 56 -z 2 -t short bs.raw bs.mrc >/dev/null
$R/mrc/raw2mrc -x 60 -y 56 -z 1 -t float bl.raw odd.mrc >/dev/null
rm -f *.raw
