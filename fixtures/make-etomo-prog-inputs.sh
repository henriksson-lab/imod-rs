#!/bin/bash
# Inputs for the golden suites of the eTomo-launched programs translated on
# 2026-10-05 (avgstack, extractpieces, goodframe, rotatevol, taperoutvol,
# remapmodel, xfjointomo).  Containers are written by the native reference
# (`raw2mrc`, `point2model`, `xfjointomo`) from seeded content, so they are
# IMOD's own files.  Run once; the outputs are committed fixtures.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
export LD_LIBRARY_PATH=$R/buildlib AUTODOC_DIR=$R/autodoc
F=$(cd "$(dirname "$0")" && pwd)
S=$(mktemp -d)
cd "$S"
python3 - <<'EOF'
import numpy as np, random, struct
r = np.random.default_rng(7)
r.normal(100, 20, (4, 24, 32)).astype(np.float32).tofile('v.raw')
r.integers(0, 30000, (3, 20, 26)).astype(np.int16).tofile('s.raw')
r.integers(400, 800, (18, 8, 8)).astype(np.int16).tofile('bm.raw')
random.seed(3)
with open('pts1.txt', 'w') as f:
    for c in range(1, 6):
        for z in range(0, 10, 1 if c % 2 else 2):
            f.write(f"1 {c} {random.uniform(5,60):.2f} {random.uniform(5,40):.2f} {z}\n")
    for c in range(1, 3):
        for z in [3, 7, 1]:
            f.write(f"2 {c} {random.uniform(5,60):.2f} {random.uniform(5,40):.2f} {z}\n")
with open('join.txt', 'w') as f:
    c = 0
    for k in range(12):
        c += 1
        x0, y0 = random.uniform(20, 200), random.uniform(20, 150)
        sx, sy = random.uniform(-.5, .5), random.uniform(-.5, .5)
        for z in range(2, 20):
            dx = 3.0 if z >= 10 else 0.0
            f.write(f"1 {c} {x0+sx*z+dx+random.gauss(0,.2):.2f} "
                    f"{y0+sy*z+1.5*(z>=10)+random.gauss(0,.2):.2f} {z}\n")
    for k in range(10):
        c += 1
        x0, y0 = random.uniform(20, 200), random.uniform(20, 150)
        sx, sy = random.uniform(-.5, .5), random.uniform(-.5, .5)
        for z in range(14, 29):
            dx = -2.0 if z >= 22 else 0.0
            f.write(f"1 {c} {x0+sx*z+dx+random.gauss(0,.2):.2f} "
                    f"{y0+sy*z+1.0*(z>=22)+random.gauss(0,.2):.2f} {z}\n")
    for k in range(5):
        x0, y0 = random.uniform(20, 200), random.uniform(20, 150)
        f.write(f"2 {k+1} {x0:.2f} {y0:.2f} 9\n2 {k+1} {x0+3.1:.2f} {y0+1.4:.2f} 10\n")
with open('thick.txt', 'w') as f:
    f.write('0.5\n0.7\n0.6\n0.65\n0.6\n0.6\n0.62\n0.6\n0.6\n0.61\n0.6\n')
EOF
$R/mrc/raw2mrc -x 32 -y 24 -z 4 -t float v.raw v.mrc > /dev/null
$R/mrc/raw2mrc -x 26 -y 20 -z 3 -t short s.raw s.mrc > /dev/null
$R/mrc/raw2mrc -x 8 -y 8 -z 18 -t short bm.raw bm.st > /dev/null
python3 - <<'EOF'
import struct
pl = [(44 * (i % 3), 40 * ((i // 3) % 3), i // 9) for i in range(18)]
d = bytearray(open('bm.st', 'rb').read())
ext = b''.join(struct.pack('<3h', *p) for p in pl)
hdr = d[:1024]
struct.pack_into('<i', hdr, 92, len(ext))
struct.pack_into('<hh', hdr, 128, 6, 2)
open('pc.st', 'wb').write(bytes(hdr) + ext + bytes(d[1024:]))
head = 'PixelSpacing = 1\nImageFile = bm.st\nImageSize = 8 8\nMontage = 1\nDataMode = 1\n\n'
with open('bm.st.mdoc', 'w') as f:
    f.write(head)
    for i, p in enumerate(pl):
        f.write('[ZValue = %d]\nPieceCoordinates = %d %d %d\nTiltAngle = 0\n\n' % (i, *p))
with open('short.mdoc', 'w') as f:
    f.write(head)
    for i, p in enumerate(pl[:5]):
        f.write('[ZValue = %d]\nPieceCoordinates = %d %d %d\n\n' % (i, *p))
EOF
$R/imodutil/point2model -input pts1.txt -output r1.mod -image v.mrc > /dev/null
$R/imodutil/point2model -input join.txt -output j1.mod -scat 0 > /dev/null
$R/flib/model/xfjointomo -input j1.mod -foutput fprev.xf -goutput gprev.xf -sizes 10,12,8 > /dev/null
for d in avgstack rotatevol taperoutvol; do mkdir -p "$F/$d"; cp v.mrc s.mrc "$F/$d/"; done
mkdir -p "$F/extractpieces" "$F/goodframe" "$F/remapmodel" "$F/xfjointomo"
cp pc.st bm.st bm.st.mdoc short.mdoc s.mrc "$F/extractpieces/"
cp r1.mod thick.txt "$F/remapmodel/"   # thick2.txt (24 values) written by hand
cp j1.mod fprev.xf "$F/xfjointomo/"
rm -rf "$S"
# imod2patch models: native patch2imod from the patch2imod suite's patch files
# (vals6/vals3/plain, and vals6 with -f).  joinwarp2model: patch2imod's
# ctrl.warp (control points) and grid.warp, plus v.mrc as a joined file.
P=$F/patch2imod
mkdir -p "$F/imod2patch" "$F/joinwarp2model"
for f in vals6 vals3 plain; do $R/imodutil/patch2imod $P/$f.out "$F/imod2patch/$f.mod" > /dev/null; done
$R/imodutil/patch2imod -f $P/vals6.out "$F/imod2patch/v6f.mod" > /dev/null
cp $P/ctrl.warp $P/grid.warp "$F/joinwarp2model/"
cp "$F/avgstack/v.mrc" "$F/joinwarp2model/"
# flattenwarp: boundary contours (two surfaces, and one), scattered points
# (two objects, one outlier each), on a 300x250x60 volume.
S2=$(mktemp -d); cd "$S2"
python3 - <<'PY'
import math, random
random.seed(11)
def surf(x,y,off): return off + 6*math.sin(x/45.0) + y/25.0 + 3*math.cos(y/30.0)
with open('bnd.txt','w') as f:
    c=0
    for iy in range(10, 240, 20):
        c+=1
        for x in range(5, 295, 15):
            f.write(f"1 {c} {x+random.uniform(-2,2):.2f} {iy} {surf(x,iy,20):.2f}\n")
        c+=1
        for x in range(10, 290, 17):
            f.write(f"1 {c} {x+random.uniform(-2,2):.2f} {iy} {surf(x,iy,40):.2f}\n")
with open('bnd1.txt','w') as f:
    c=0
    for iy in range(10, 240, 20):
        c+=1
        for x in range(5, 295, 15):
            f.write(f"1 {c} {x+random.uniform(-2,2):.2f} {iy} {surf(x,iy,30):.2f}\n")
with open('scat.txt','w') as f:
    for ob,off in [(1,20),(2,42)]:
        for k in range(90):
            x=random.uniform(10,290); y=random.uniform(10,240)
            z=surf(x,y,off)+random.gauss(0,0.3)
            if k==5: z+=8
            f.write(f"{ob} 1 {x:.2f} {y:.2f} {z:.2f}\n")
PY
mkdir -p "$F/flattenwarp"
$R/imodutil/point2model -input bnd.txt -output "$F/flattenwarp/bnd.mod" -open -volume 300,250,60 > /dev/null
$R/imodutil/point2model -input bnd1.txt -output "$F/flattenwarp/bnd1.mod" -open -volume 300,250,60 > /dev/null
$R/imodutil/point2model -input scat.txt -output "$F/flattenwarp/scat.mod" -scat -volume 300,250,60 > /dev/null
cd /; rm -rf "$S2"
# nad_eed_3d: a 24x20x8 float and short volume with structure and noise.
S3=$(mktemp -d); cd "$S3"
python3 -c "
import numpy as np
r=np.random.default_rng(5)
z,y,x=np.mgrid[0:8,0:20,0:24]
v=(100+40*np.sin(x/4.0)*np.cos(y/5.0)+20*(z>3)+r.normal(0,5,(8,20,24))).astype(np.float32); v.tofile('v.raw')
v.astype(np.int16).tofile('s.raw')
"
mkdir -p "$F/nad_eed_3d"
$R/mrc/raw2mrc -x 24 -y 20 -z 8 -t float v.raw "$F/nad_eed_3d/v.mrc" > /dev/null
$R/mrc/raw2mrc -x 24 -y 20 -z 8 -t short s.raw "$F/nad_eed_3d/s.mrc" > /dev/null
cd /; rm -rf "$S3"
# subimage: v.mrc, v2.mrc (v.mrc plus seeded noise), s.mrc and a `v.mrc~`
# copy (file B's default).  maxjoinsize: three float tomograms of different
# sizes (t0..t2), an .info listing them and a .tomoxg of transforms (j), and
# patch2imod's grid.warp as a warping .tomoxg (w).  boxstartend: a 64x48x5
# volume and a model of 4 open contours and one 2-point object made with
# point2model -image.  model2point/repackseed: the remapmodel and xfjointomo
# models, a mesh made with the native imodmesh -c, imod2patch's vals6.mod,
# flattenwarp's scat.mod, and an xyz list (point obj cont).  imodauto: the
# avgstack volumes.  These were made in /big/henriksson/extwork/fx with the
# commands recorded in TODO.md (2026-10-05) and copied here.
