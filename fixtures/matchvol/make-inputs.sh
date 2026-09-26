#!/bin/bash
# Regenerates the seeded matchvol/warpvol inputs in this directory with the native
# raw2mrc (so IMOD itself writes the MRC containers), plus the 3x4 transform files
# and, for warpvol, findwarp-format warp files built from a known smooth field.
# Usage: make-inputs.sh [matchvol|warpvol]  (default: matchvol)
# The volumes are made only for matchvol; warpvol reads fixtures/matchvol's
# vf.mrc, vs.mrc and mb.mrc.
set -e
cd "$(dirname "$0")/../${1:-matchvol}"
R=${IMOD_REF:-/tmp/imod-reference-build}
python3 - <<'PY'
import numpy as np
def vol(nx,ny,nz,seed):
    r=np.random.default_rng(seed)
    z,y,x=np.mgrid[0:nz,0:ny,0:nx].astype(np.float32)
    v=np.zeros((nz,ny,nx),np.float32)
    for k in range(10):
        cx,cy,cz=r.uniform(0,nx),r.uniform(0,ny),r.uniform(0,nz)
        s=r.uniform(1.5,max(2.5,nx/6))
        v+=r.uniform(-60,100)*np.exp(-((x-cx)**2+(y-cy)**2+(z-cz)**2)/(2*s*s))
    return v+0.3*x-0.2*y+0.5*z+r.normal(0,4,v.shape)
vf=vol(25,19,13,1).astype(np.float32)
vs=(vol(21,17,11,2)*20).astype(np.int16)
mb=np.clip(vol(100,90,30,3)+100,0,255).astype(np.uint8)
import os
if os.path.basename(os.getcwd())=='matchvol':
    vf.tofile("vf.raw"); vs.tofile("vs.raw"); mb.tofile("mb.raw")
def rot(ax,deg):
    t=np.radians(deg);c,s=np.cos(t),np.sin(t)
    if ax=='x': return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if ax=='y': return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def wr(name,A,d):
    with open(name,'w') as f:
        for i in range(3): f.write("%12.7f%12.7f%12.7f%12.3f\n"%(A[i,0],A[i,1],A[i,2],d[i]))
import os
if os.path.basename(os.getcwd())=='matchvol':
    wr("t1.xf",1.02*rot('y',12)@rot('x',3),[2.5,-1.3,0.7])
    wr("t2.xf",rot('z',-4)@rot('y',-2),[-1.1,0.4,3.2])
    wr("tyx.xf",rot('z',88),[0.3,0.2,-0.5])
    wr("tzx.xf",rot('y',92)@rot('x',5),[1.0,-2.0,0.5])
    wr("tflip.xf",np.diag([-1,1,-1])@rot('y',3),[0.0,0.0,0.0])
else:
    def field(x,y,z,flip):
        A=rot('y',3+0.05*x)@rot('x',-2+0.04*z)@rot('z',1+0.03*y)*(1+0.001*z)
        d=np.array([1.5+0.03*x-0.02*z,-0.8+0.02*y,0.6+0.04*x])
        if flip: A=np.array([[0,0,1],[0,1,0],[-1,0,0]])@A
        return A,d
    def write(name,xs,ys,zs,kind=9,drop=(),flip=False,rev=False):
        pos=[(x,y,z) for z in zs for y in ys for x in xs]
        with open(name,'w') as f:
            if kind==2: f.write("%d %d\n"%(len(xs),len(zs)))
            elif kind==3: f.write("%d %d %d\n"%(len(xs),len(ys),len(zs)))
            else:
                step=lambda a:(a[1]-a[0]) if len(a)>1 else 1
                f.write("%5d%6d%6d%11.2f%11.2f%11.2f%10.4f%10.4f%10.4f\n"%(len(xs),len(ys),len(zs),xs[0],ys[0],zs[0],step(xs),step(ys),step(zs)))
            idx=list(range(len(pos)))
            if rev: idx=idx[::-1]
            for i in idx:
                if i in drop: continue
                x,y,z=pos[i]; A,d=field(x,y,z,flip)
                f.write("%9.1f%9.1f%9.1f\n"%(x,y,z))
                for r in range(3): f.write("%11.6f%11.6f%11.6f%11.3f\n"%(A[r,0],A[r,1],A[r,2],d[r]))
    xs=[-9,-3,3,9]; ys=[-6,0,6]; zs=[-4,0,4]
    write("w9.txt",xs,ys,zs)
    write("w3.txt",xs,ys,zs,kind=3,rev=True)
    write("wdrop.txt",xs,ys,zs,drop=(0,1,5,12,13,30,35))
    write("w2d.txt",[-10,-3,3,10],[0],[-5,0,5],kind=2)
    write("w1y.txt",xs,[0],zs)
    write("wsmall.txt",[-4,4],[-3,3],[-2,2])
    write("wflip.txt",[-4,0,4],[-6,0,6],[-9,0,9],flip=True)
    write("wbad.txt",xs,ys,zs,kind=3,drop=(3,))
    write("wmb.txt",[-40,-10,20,40],[-35,0,35],[-14,0,14],drop=(2,20))
    open("w5.txt","w").write("4 3 3 1 2\n")
    open("wempty.txt","w").write("")
PY
export LD_LIBRARY_PATH=$R/buildlib
M=$R/mrc/raw2mrc
if [ -f mb.raw ]; then
  rm -f vf.mrc vs.mrc mb.mrc  # else raw2mrc leaves `~` backups
  $M -x 25 -y 19 -z 13 -t float vf.raw vf.mrc > /dev/null
  $M -x 21 -y 17 -z 11 -t short vs.raw vs.mrc > /dev/null
  $M -x 100 -y 90 -z 30 -t byte mb.raw mb.mrc > /dev/null
fi
rm -f *.raw
