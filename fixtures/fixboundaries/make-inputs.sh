#!/bin/bash
# Regenerates the fixboundaries inputs with the native raw2mrc: main files
# (float and byte) and boundary (guard) files, seeded, and boundary-info files
# in the format parallelwrite.c reads (version, chunks-in-Y flag, nx, lines per
# boundary, number of files; then per file its name and
# "section1 startLine1 section2 startLine2").
set -e
cd "$(dirname "$0")"
R=${IMOD_REF:-/tmp/imod-reference-build}
python3 - <<'PY'
import numpy as np
rng=np.random.default_rng(5)
nx,ny,nz=20,16,6
rng.normal(0,1,(nz,ny,nx)).astype(np.float32).tofile("mainf.raw")
(rng.random((nz,ny,nx))*50).astype(np.uint8).tofile("mainb.raw")
# chunks in Z: two sections of 3 lines each
for k in range(3):
    (rng.normal(100+k,5,(2,3,nx))).astype(np.float32).tofile(f"bz{k}.raw")
# chunks in Y: 2 * nz sections of 3 lines each
for k in range(2):
    (rng.normal(200+k,5,(2*nz,3,nx))).astype(np.float32).tofile(f"by{k}.raw")
(rng.normal(9,1,(1,2,nx))).astype(np.float32).tofile("short.raw")
open("z.info","w").write("1 0 20 3 3\nbz0.mrc\n-1 0 1 -1\nbz1.mrc\n1 0 3 -1\nbz2.mrc\n3 0 -1 0\n")
open("z2.info","w").write("1 0 20 3 1\nbz0.mrc\n2 5 4 7\n")
open("y.info","w").write("1 1 20 3 2\nby0.mrc\n-1 -1 -1 4\nby1.mrc\n-1 9 -1 -1\n")
open("badnx.info","w").write("1 0 21 3 1\nbz0.mrc\n1 0 2 0\n")
open("nolines.info","w").write("1 0 20 0 1\nbz0.mrc\n1 0 2 0\n")
open("trunc.info","w").write("1 0 20 3 2\nbz0.mrc\n1 0 2 0\n")
open("nofile.info","w").write("1 0 20 3 1\nnosuch.mrc\n1 0 2 0\n")
open("short.info","w").write("1 0 20 3 1\nshort.mrc\n1 0 2 0\n")
open("empty.info","w").write("")
PY
export LD_LIBRARY_PATH=$R/buildlib
$R/mrc/raw2mrc -x 20 -y 16 -z 6 -t float mainf.raw mainf.mrc >/dev/null
$R/mrc/raw2mrc -x 20 -y 16 -z 6 -t byte mainb.raw mainb.mrc >/dev/null
for k in 0 1 2; do $R/mrc/raw2mrc -x 20 -y 3 -z 2 -t float bz$k.raw bz$k.mrc >/dev/null; done
for k in 0 1; do $R/mrc/raw2mrc -x 20 -y 3 -z 12 -t float by$k.raw by$k.mrc >/dev/null; done
$R/mrc/raw2mrc -x 20 -y 2 -z 1 -t float short.raw short.mrc >/dev/null
rm -f *.raw
