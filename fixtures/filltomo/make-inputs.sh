#!/bin/bash
# Regenerates the seeded filltomo inputs in this directory with the native
# raw2mrc, matchvol (the transformed volume to fill and its inverse
# transform) and point2model (boundary models).
set -e
cd "$(dirname "$0")"
R=${IMOD_REF:-/tmp/imod-reference-build}
export LD_LIBRARY_PATH=$R/buildlib AUTODOC_DIR=$R/autodoc
python3 - <<'PY'
import numpy as np
rng=np.random.default_rng(5)
nx,ny,nz=19,9,15
z,y,x=np.mgrid[0:nz,0:ny,0:nx]
b=(np.sin(x/3.0)*np.cos(z/4.0)+0.05*y)*40+100+rng.normal(0,2,(nz,ny,nx))
b.astype(np.float32).tofile("src.raw")
a=(np.sin(x/3.0+0.2)*np.cos(z/4.0)+0.05*y)*30+50+rng.normal(0,2,(nz,ny,nx))
a.astype(np.float32).tofile("mat.raw")
(a*3).astype(np.int16).tofile("mats.raw")
np.arange(10*9*15,dtype=np.float32).tofile("other.raw")
with open("stack.xf","w") as f:
  for i in range(16):
    t=np.radians(3+0.1*i); c,s=np.cos(t),np.sin(t)
    f.write("%12.7f%12.7f%12.7f%12.7f%12.3f%12.3f\n"%(1.01*c,-s,s,0.99*c,1.5+i*0.1,-2.0))
with open("stackfew.xf","w") as f:
  for i in range(3):
    f.write("%12.7f%12.7f%12.7f%12.7f%12.3f%12.3f\n"%(1,0.05,-0.05,1,0,0))
def w(n,conts):
  with open(n,"w") as f:
    for c,pts in enumerate(conts):
      for p in pts: f.write("1 %d %g %g %g\n"%((c+1,)+p))
w("bnd.txt",[[(2,4,1),(16,4,2),(17,4,13),(3,4,14)],[(1,7,1),(18,7,1),(18,7,14),(1,7,14)]])
w("bnd2.txt",[[(4,5,2),(15,5,3),(16,5,12),(3,5,11),(1,5,6)]])
w("bndxy.txt",[[(3,2,3),(15,7,3),(16,2,3)],[(3,2,5),(15,7,5),(16,2,5)]])
w("mix.txt",[[(3,2,3),(15,7,7),(16,2,9)]])
PY
M=$R/mrc/raw2mrc
$M -x 19 -y 9 -z 15 -t float src.raw src.mrc >/dev/null
$M -x 19 -y 9 -z 15 -t float mat.raw mat.mrc >/dev/null
$M -x 19 -y 9 -z 15 -t short mats.raw mats.mrc >/dev/null
$M -x 10 -y 9 -z 15 -t float other.raw other.mrc >/dev/null
$R/flib/image/matchvol -input src.mrc -output fill.mrc -3dxform 0.98,0.1,0.05,1.5,-0.1,0.99,0.02,-0.7,-0.04,-0.03,1,2.2 -inverse inv.xf >/dev/null
$R/flib/image/matchvol -input src.mrc -output fills.mrc -3dxform 0.9,0.2,0.1,3,-0.2,0.95,0,-1,-0.1,0,0.9,2 -inverse inv2.xf >/dev/null
for b in bnd bnd2 bndxy mix; do $R/imodutil/point2model -image src.mrc $b.txt $b.mod >/dev/null; done
rm -f *.raw *.txt *~
