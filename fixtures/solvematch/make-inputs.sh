#!/bin/bash
# Regenerate the solvematch fixture inputs: synthetic dual-axis bead sets from
# make-inputs.py (seed 7; B related to A by a ~91.5 degree rotation about the
# depth axis plus a small tilt and shift), tiltalign-format point files and
# variants derived from them, 2D fiducial and matching models written by the
# native point2model, and two tiny tomograms (pixel 1 and 2) from native
# raw2mrc + alterheader.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
export AUTODOC_DIR=$R/autodoc LD_LIBRARY_PATH=$R/buildlib
F=$(cd "$(dirname "$0")" && pwd)
S=$(mktemp -d)
python3 "$F/make-inputs.py" 7 "$S" >/dev/null
cd "$S"
sed '1s/ Pix:.*$//' twoA.xyz > relA.xyz; sed '1s/ Pix:.*$//' twoB.xyz > relB.xyz
sed '1s/ Pix: *\([0-9.]*\) Dim:.*$/ Pixel size: \1/' twoA.xyz > oldA.xyz
sed '1s/ Pix: *\([0-9.]*\) Dim:.*$/ Pixel size: \1/' twoB.xyz > oldB.xyz
: > empty.xyz
sed '3s/.*/   3    abc  1 2 1 3/' twoA.xyz > badA.xyz
sed '1s/Dim: *\([0-9]*\) *[0-9]*/Dim: \1 x/' twoA.xyz > baddimA.xyz
head -3 twoA.xyz > fewA.xyz; head -3 twoB.xyz > fewB.xyz
printf '   1    10.0 20.0 3.0 1 1\n\n   2 5 5 5 1 2\n' > blankline.xyz
awk 'NR==2{$1=0} {print}' twoA.xyz > zeroidA.xyz
awk 'NR==1{print "21 22 0"; next}{print}' two.transfer > tr_badz.transfer
awk 'NR==1{print "22 20 1"; next}{print $3, $4, $1, $2}' two.transfer > tr_btoa.transfer
awk 'NR==1{print "20 22 0 2.0"; next}{printf "%.2f %.2f %.2f %.2f\n", $1*0.5,$2*0.5,$3*0.5,$4*0.5}' \
    two.transfer > tr_pix.transfer
awk 'NR==3{$3=$3+4; $5=$5-3} {print}' twoBmatch.txt > twoBmatchbad.txt
python3 -c "
import random
random.seed(5)
for l in open('twoBmatch.txt'):
    f=l.split(); print(f[0],f[1],' '.join('%.3f'%(float(v)+random.gauss(0,2.5)) for v in f[2:]))" \
    > twoBmatchnoisy.txt
for b in twoAmatch twoBmatch oneBmatch twoAfid twoBfid oneAfid oneBfid twoBmatchbad twoBmatchnoisy \
    warpAfid warpBfid; do
  $R/imodutil/point2model -op $b.txt $b.mod >/dev/null
done
python3 -c "import numpy as np; np.zeros((10,5,10),np.uint8).tofile('z.raw')"
$R/mrc/raw2mrc -x 10 -y 5 -z 10 -t byte z.raw tiny1.mrc >/dev/null
cp tiny1.mrc tiny2.mrc; $R/flib/image/alterheader -del 2,2,2 tiny2.mrc >/dev/null
python3 -c "import numpy as np; np.zeros((10,8,10),np.uint8).tofile('z.raw')"
$R/mrc/raw2mrc -x 10 -y 8 -z 10 -t byte z.raw tiny3.mrc >/dev/null
C2=$(cat two.corr)
printf 'AFiducialFile twoA.xyz\nBFiducialFile twoB.xyz\nACorrespondenceList 1-30\n' > params.txt
printf 'BCorrespondenceList %s\nATomogramOrSizeXYZ tiny1.mrc\n' $C2 >> params.txt
printf 'BTomogramOrSizeXYZ 400,100,400\nOutputFile out.xf\n' >> params.txt
for f in twoA.xyz twoB.xyz oneA.xyz oneB.xyz invA.xyz invB.xyz outA.xyz outB.xyz nocolA.xyz \
    nocolB.xyz relA.xyz relB.xyz oldA.xyz oldB.xyz badA.xyz baddimA.xyz empty.xyz fewA.xyz \
    fewB.xyz blankline.xyz zeroidA.xyz twoAmatch.mod twoBmatch.mod oneBmatch.mod twoAfid.mod \
    twoBfid.mod oneAfid.mod oneBfid.mod two.transfer tr_btoa.transfer tr_pix.transfer \
    tr_badz.transfer tiny1.mrc tiny2.mrc tiny3.mrc twoBmatchbad.mod twoBmatchnoisy.mod warpA.xyz \
    warpB.xyz warp.transfer warpAfid.mod warpBfid.mod hiidA.xyz hiidB.xyz sclA.xyz sclB.xyz \
    params.txt; do
  cp $f "$F/"
done
rm -rf "$S"
