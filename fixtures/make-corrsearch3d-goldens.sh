#!/bin/bash
# Regenerate fixtures/corrsearch3d/ (inputs and golden/) from the native reference.
# Inputs are authored here: two small seeded synthetic dual-axis-like volume
# pairs (a smoothed random structure plus point "beads" in a slab, B = A
# resampled through a smooth known shift field plus independent noise),
# written by native raw2mrc as bytes -- s1 unflipped 48x48x20, s2 flipped
# 45x19x43 with B in a bigger box -- and boundary models written by
# native point2model.  For each row of cases.tsv the native program runs in a
# fresh directory (OMP_NUM_THREADS=1) with stdout captured through a pipe;
# golden/<name>.rc is the exit status, .stdout the standard output and
# .patch the displacement file, when native leaves one.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
F=$(cd "$(dirname "$0")/corrsearch3d" && pwd)
export AUTODOC_DIR=$R/autodoc LD_LIBRARY_PATH=$R/buildlib
S=$(mktemp -d)
cd "$S"
python3 - <<'PY'
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
def make(tag, nx, ny, nz, seed, flipped, dtype, bpad):
    r = np.random.default_rng(seed)
    v = gaussian_filter(r.standard_normal((nz, ny, nx)), 2.0) * 6
    pts = np.zeros_like(v)
    n = nx * ny * nz // 1500
    pts[r.integers(0, nz, n), r.integers(0, ny, n), r.integers(0, nx, n)] = r.uniform(20, 60, n)
    v += gaussian_filter(pts, 1.2)
    zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32)
    t = (yy - ny / 2) / (0.35 * ny) if flipped else (zz - nz / 2) / (0.35 * nz)
    v *= 1 / (1 + np.exp((np.abs(t) - 1) * 8))
    f = [gaussian_filter(r.standard_normal(v.shape), 8) for _ in range(3)]
    f = [g / np.abs(g).max() * 2 + s for g, s in zip(f, (1.2, -0.6, 0.8))]
    a = v + 0.3 * r.standard_normal(v.shape)
    b = map_coordinates(v, [zz - f[2], yy - f[1], xx - f[0]], order=1, mode='nearest')
    b = b + 0.3 * r.standard_normal(v.shape)
    if bpad:
        b = np.pad(b, ((1, 1), (2, 2), (3, 3)), mode='edge')
    for k, arr in (('a', a), ('b', b)):
        if dtype == 'byte':
            arr = np.clip((arr + 6) * 20, 0, 255).astype(np.uint8)
        elif dtype == 'short':
            arr = np.clip(arr * 800, -32000, 32000).astype(np.int16)
        else:
            arr = arr.astype(np.float32)
        arr.tofile(f'{tag}_{k}.raw')
        open(f'{tag}_{k}.dim', 'w').write('%d %d %d %s\n' % (arr.shape[2], arr.shape[1], arr.shape[0], dtype))
make('s1', 48, 48, 20, 101, False, 'byte', False)
make('s2', 45, 19, 43, 102, True, 'byte', True)
PY
for f in *.raw; do
  read x y z t < ${f%.raw}.dim
  $R/mrc/raw2mrc -x $x -y $y -z $z -t $t $f ${f%.raw}.mrc > /dev/null
done
printf "1 1 6 6 5\n1 1 42 8 5\n1 1 40 40 5\n1 1 8 42 5\n1 2 10 4 15\n1 2 44 10 15\n1 2 40 44 15\n1 2 4 40 15\n" > rega.txt
$R/imodutil/point2model -image s1_a.mrc rega.txt s1_rega.mod > /dev/null
printf "1 1 2 14 3\n1 1 46 14 3\n1 1 46 46 3\n1 1 2 46 3\n" > regb.txt
$R/imodutil/point2model -image s1_b.mrc regb.txt s1_regb.mod > /dev/null
printf "1 1 6 6 5\n1 1 42 8 7\n1 1 40 40 9\n" > nonpl.txt
$R/imodutil/point2model -image s1_a.mrc nonpl.txt s1_nonplanar.mod > /dev/null
printf "1 1 4 9 4\n1 1 41 9 6\n1 1 40 9 38\n1 1 6 9 36\n" > reg2.txt
$R/imodutil/point2model -image s2_a.mrc reg2.txt s2_rega.mod > /dev/null
printf "1 1 20 20 5\n1 1 30 20 5\n1 1 30 30 5\n1 1 20 30 5\n" > small.txt
$R/imodutil/point2model -image s1_a.mrc small.txt s1_small.mod > /dev/null
printf "0.9998 0.0175 0.002 1.5\n-0.0175 0.9998 0.001 -2.0\n-0.002 -0.001 1.0 0.5\n" > small.xf
printf "0.99 0.1 0.0 3\n-0.1 0.99 0.02 1\n" > short.xf
printf "ReferenceFile s1_a.mrc\nFileToAlign s1_b.mrc\nOutputFile out.patch\nPatchSizeXYZ 14,14,8\nNumberOfPatchesXYZ 3,3,2\nXMinAndMax 4,45\nYMinAndMax 4,45\nZMinAndMax 2,19\nBSourceBorderXLoHi 3,3\nBSourceBorderYZLoHi 2,2\nKernelSigma 1.2\n" > cs.param
rm -f $F/*.mrc $F/*.mod $F/*.xf $F/*.param
cp s1_a.mrc s1_b.mrc s2_a.mrc s2_b.mrc s1_rega.mod s1_regb.mod s1_nonplanar.mod s2_rega.mod s1_small.mod small.xf short.xf cs.param $F/
rm -rf $F/golden; mkdir -p $F/golden
# An optional fourth column is a sed expression applied to the native stdout
# of a case that reaches an upstream bug fixed in the translation (BUGS.md).
while IFS=$'\t' read -r name args stdin sub; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  d=$S/run-$name; rm -rf $d; mkdir $d; cp $F/*.mrc $F/*.mod $F/*.xf $F/*.param $d/; cd $d
  [ "$args" = "-" ] && args=""
  set +e
  if [ "$stdin" = "-" ] || [ -z "$stdin" ]; then
    OMP_NUM_THREADS=1 $R/flib/model/corrsearch3d $args < /dev/null 2>/dev/null | cat > $F/golden/$name.stdout
    echo ${PIPESTATUS[0]} > $F/golden/$name.rc
  else
    printf "$stdin" | OMP_NUM_THREADS=1 $R/flib/model/corrsearch3d $args 2>/dev/null | cat > $F/golden/$name.stdout
    echo ${PIPESTATUS[1]} > $F/golden/$name.rc
  fi
  set -e
  if [ -n "$sub" ] && [ "$sub" != "-" ]; then sed -i -e "$sub" $F/golden/$name.stdout; fi
  [ -e out.patch ] && cp out.patch $F/golden/$name.patch
  cd $S
done < <(sed "${FULL:+s/^#full\t//;}/^#/d" $F/cases.tsv)
rm -rf "$S"
echo "goldens: $(ls $F/golden/*.rc | wc -l) cases"
