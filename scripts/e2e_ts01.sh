#!/bin/bash
# End-to-end single-axis tomography pipeline differential on REAL data:
# native IMOD vs imod-rs, step by step.  Results are described in E2E.md.
#
#   SRC     the tilt series (read-only; it is only ever read, never written)
#   B       scratch root on /big (needs ~6 GB)
#   PHASES  any of: pipe cross omp1       (default: all three)
#   ONLY    extended regex; run only steps whose name matches (pipe/cross/omp1)
#
# Phases
#   pipe   each side runs every step in its own directory, consuming ITS OWN
#          previous outputs (a true pipeline); outputs are compared per step.
#   cross  imod-rs runs each step on NATIVE's previous outputs, so a
#          difference there belongs to that step alone.  A step whose inputs
#          all come from setup is identical to its pipe run and is not repeated.
#   omp1   the pipe phase again with OMP_NUM_THREADS=1 on both sides, for the
#          single-threaded timing; its outputs are compared against the default
#          pipe run of the same side (they should not depend on thread count).
#
# Each command is invoked the way the IMOD .com templates (IMOD/com/*.com and
# the files copytomocoms/batchruntomo write from them) invoke it: parameter
# lines on stdin to `-StandardInput`, or xftoxg's interactive answers.
#
# The true tilt angles of TS_01 are not in its header; -51..+51 step 3 is
# ASSUMED.  The tilt-axis rotation is set to -90 because tiltalign, run at 0,
# reported "the rotation angle seems to be closer to -90.0".
#  This is an equivalence test between two implementations, not a
# scientific reconstruction.
#
# Steps 23-46 (2026-09-26) add the setup scripts, erasing, findbeads3d, CTF
# correction, MTF/dose filtering, tomopitch, xcorrstack, extracttilts and
# splittilt/processchunks; steps 50-67 a dual-axis combine on a SYNTHETIC B
# axis built from the native pipeline's outputs (dual_setup).  Several inputs
# are SYNTHETIC or ASSUMED (see the data section and E2E.md).  Do not edit
# this file while it runs: bash reads it incrementally; run a copy with
# RSBIN= set instead.
export LC_ALL=C

SRC=${SRC:-/husky/otherdataset/teresa/EM/4johan/TS_01.mrc}
REF=${REF:-/tmp/imod-reference-build}
REPO=$(cd "$(dirname "$0")/.." && pwd)
RSBIN=${RSBIN:-$REPO/target/release/imod}
B=${B:-/big/henriksson/realbench/e2e-ts01}
PHASES=${PHASES:-pipe cross omp1}
ONLY=${ONLY:-}

[ -d $REF/flib ] || ln -sfn /big/henriksson/imod-reference-build /tmp/imod-reference-build
[ -x $RSBIN ] || { echo "build first: cargo build --release --bin imod" >&2; exit 1; }
mkdir -p $B/data $B/steps

# ------------------------------------------------------------------ installs
# Rust install: one link per command in src/imod/commands.rs (read from the
# launcher's own usage listing, which prints that table), all -> imod.
# The binary is COPIED first: target/release/imod may be rebuilt while the
# pipeline runs, and every step must use the same build.
RSI=$B/rsimod; rm -rf $RSI; mkdir -p $RSI/bin
cp $RSBIN $RSI/imod; RSBIN=$RSI/imod
echo "imod-rs binary: $(sha256sum < $RSBIN | cut -c1-16) ($(cd $REPO && git rev-parse --short HEAD) + worktree)" >&2
RSCMDS=$($RSBIN 2>&1 | sed -n '/^Commands:/,$p' | tail -n +2 | awk '{print $1}' | tr '\n' ' ')
for c in $RSCMDS; do ln -s $RSBIN $RSI/bin/$c; done
# Native install: the same names, pointing at the reference build's binaries,
# plus pylib for the Python scripts (trimvol).  Anything not in the Rust table
# is still linked so a native script can find it, but no step uses it.
NTI=$B/natimod; rm -rf $NTI; mkdir -p $NTI/bin
ln -s $REF/pysrc $NTI/pylib
for d in flib/image flib/model flib/tilt flib/tiltalign flib/beadtrack flib/blend \
         flib/distort imodutil clip mrc qttools/mrc2tif qttools/processchunks pysrc \
         scripts; do
  for f in $REF/$d/*; do
    n=$(basename $f)
    [ -f $f ] && [ -x $f ] && [[ $n != *.* ]] && [ ! -e $NTI/bin/$n ] && ln -s $f $NTI/bin/$n
  done
done
# Both sides read the .com templates from $IMOD_DIR/com (copytomocoms,
# setupcombine); they are the same vendored files.
ln -s $REF/com $NTI/com; ln -s $REF/com $RSI/com
# processchunks converts each chunk .com with vmstopy, an external process
# boundary in the translation too (CLAUDE.md); the Rust side gets the native
# Python script and the pylib it imports.  Nothing else on that side is Python.
ln -s $REF/pysrc/vmstopy $RSI/bin/vmstopy; ln -s $REF/pysrc $RSI/pylib
# Every helper the translated scripts call is an imod-rs command now
# (fixboundaries, tomopieces, xyzproj and matchrotpairs were run natively on
# this side until 2026-09-26; see E2E.md).
# comrun X.com runs a command file the way combine.com / etomo run one,
#   vmstocsh X.log < X.com | csh -ef
# then prints the log (the programs' output goes there, not to stdout).
# vmstocsh is only a text converter and is not in the imod-rs command table,
# so both sides use the native one; every program the .com runs comes from
# the side's own PATH.  "Shell PID" lines are dropped (they are the PID).
for I in $NTI $RSI; do
  cat > $I/bin/comrun <<EOF
#!/bin/bash
set -o pipefail
c=\$1; log=\${c%.com}.log
LD_LIBRARY_PATH=$REF/buildlib $REF/flib/image/vmstocsh \$log < \$c | tcsh -ef | sed '/^Shell PID:/d'
rc=\$?
[ -f \$log ] && cat \$log
exit \$rc
EOF
  chmod +x $I/bin/comrun
done

# ---------------------------------------------------------------- input data
ln -sfn "$SRC" $B/data/TS_01.mrc
python3 - $B/data/ts.rawtlt <<'PY'
import sys
with open(sys.argv[1], 'w') as f:
    for a in range(-51, 52, 3):
        f.write('%7.2f\n' % a)
PY
# SYNTHETIC inputs for the steps added on 2026-09-26 (TS_01 carries no
# metadata, no CTF fit and no positioning model); native and imod-rs read the
# same files.
#  ts.st.mdoc   SerialEM-style metadata: the assumed angles, a dose-symmetric
#               order from 0 deg, 3 e/A^2 per image (mtffilter dose weighting,
#               extracttilts -mdoc)
#  ts.defocus   ASSUMED constant 3 um defocus, ctfplotter version-2 format
python3 - $B/data <<'PY'
import sys
d = sys.argv[1]
angles = [-51 + 3 * i for i in range(35)]
order = sorted(range(35), key=lambda i: (abs(angles[i]), angles[i] < 0))
rank = {v: k for k, v in enumerate(order)}
L = ['PixelSpacing = 4.716', 'Voltage = 300', 'ImageFile = ts.st', 'ImageSize = 1024 1024',
     'DataMode = 2', '', '[T = SerialEM: SYNTHETIC metadata for the imod-rs e2e differential]', '']
for i, a in enumerate(angles):
    r = rank[i]
    L += ['[ZValue = %d]' % i, 'TiltAngle = %.2f' % a,
          'StagePosition = %.3f %.3f' % (12.5 + 0.01 * i, -40.25), 'Magnification = 33000',
          'Intensity = 0.1234', 'ExposureDose = 3', 'PriorRecordDose = %d' % (3 * r),
          'PixelSpacing = 4.716', 'Defocus = %.4f' % (-3.0 + 0.01 * (i % 5)),
          'ExposureTime = 0.5', 'DateTime = 26-Sep-26  10:%02d:%02d' % (r * 2, r * 7 % 60), '']
open(d + '/ts.st.mdoc', 'w').write('\n'.join(L))
with open(d + '/ts.defocus', 'w') as f:
    for i, a in enumerate(angles):
        f.write('%d\t%d\t%.2f\t%.2f\t%.1f%s\n' % (i + 1, i + 1, a, a, 3000.0, '\t2' if i == 0 else ''))
# tomopitch model: two lines (top, bottom of a slab) in each of three samples,
# one sample per contour time, as drawn on top/mid/bot sample tomograms
L = []; c = 0
for t, (dt, db) in enumerate([(4, -3), (0, 0), (-5, 6)]):
    c += 1; L += ['1 %d 100 %d 0 %d' % (c, 90 + dt, t + 1), '1 %d 924 %d 0' % (c, 105 - dt)]
    c += 1; L += ['1 %d 100 %d 0 %d' % (c, 215 + db, t + 1), '1 %d 924 %d 0' % (c, 205 + db)]
open(d + '/tomopitch.txt', 'w').write('\n'.join(L) + '\n')
PY
( cd $B/data && LD_LIBRARY_PATH=$REF/buildlib $REF/imodutil/point2model -times \
    -volume 1024,300,1 tomopitch.txt tomopitch.mod > /dev/null )

# SYNTHETIC dual-axis data set (there is no real B axis for TS_01).  Built
# once, natively, from the native single-axis pipeline's outputs:
#  ga.ali   native ts.ali (09_newst) binned by 2 -> 512 x 512 x 35
#  gb.ali   native ts.rec (11_tilt) binned by 2, rotated EXACTLY 90 deg about
#           the specimen Z axis (file Y; a transpose, no interpolation),
#           rolled by (+6 X, -4 Z), reprojected by native tilt at ts.tlt's
#           angles, then mapped through exp() to ga.ali's log-intensity mean
#           and SD so tilt's LOG 0 treats both axes alike
#  g?.tlt / g?.rawtlt = native ts.tlt;  g?.xtilt = zeros;  g?.st -> g?.ali
#  ga.matmod / gb.matmod  matching models, 8 corresponding points related by
#           that exact rotation + shift (solvematch -2, matching models only)
dual_setup() {
  local P=$B/pipe/nat D=$B/data t
  for t in 09_newst/ts.ali 11_tilt/ts.rec 07_align/ts.tlt; do
    [ -f $P/$t ] || { echo "dual setup needs native $P/$t (run the pipe phase)" >&2; exit 1; }
  done
  ( cd $D
    export LD_LIBRARY_PATH=$REF/buildlib AUTODOC_DIR=$REF/autodoc
    $REF/flib/image/newstack -bin 2 $P/09_newst/ts.ali ga.ali > dualsetup.log
    $REF/flib/image/binvol -binning 2 $P/11_tilt/ts.rec a2.rec >> dualsetup.log
    cp $P/07_align/ts.tlt ga.tlt; cp ga.tlt gb.tlt; cp ga.tlt ga.rawtlt; cp ga.tlt gb.rawtlt
    for x in a b; do python3 -c "print('\n'.join(['0.00'] * 35))" > g$x.xtilt; done
    python3 - <<'PY'
import numpy as np
b = open('a2.rec', 'rb').read()
nx, ny, nz = np.frombuffer(b[:12], '<i4'); off = 1024 + np.frombuffer(b[92:96], '<i4')[0]
v = np.frombuffer(b, np.float32, nx * ny * nz, off).reshape(nz, ny, nx)
r = np.roll(np.rot90(v, 1, axes=(0, 2)), (-4, 6), axis=(0, 2))
open('b2rot.rec', 'wb').write(b[:off] + np.ascontiguousarray(r, np.float32).tobytes())
PY
    $REF/flib/tilt/tilt -StandardInput >> dualsetup.log <<EOF
InputProjections ga.ali
OutputFile gb_reproj.mrc
IMAGEBINNED 1
TILTFILE gb.tlt
FULLIMAGE 512 512
SUBSETSTART 0 0
PERPENDICULAR
THICKNESS 150
RecFileToReproject b2rot.rec
EOF
    python3 - <<'PY'
import numpy as np
def rd(p):
    b = open(p, 'rb').read(); nx, ny, nz = np.frombuffer(b[:12], '<i4')
    off = 1024 + np.frombuffer(b[92:96], '<i4')[0]
    return b, np.frombuffer(b, np.float32, nx * ny * nz, off).astype(np.float64), off
_, a, _ = rd('ga.ali'); bb, r, off = rd('gb_reproj.mrc')
la = np.log(a); c = la.std() / r.std()
g = np.exp(la.mean() + c * (r - r.mean())).astype(np.float32)
open('gb_raw.ali', 'wb').write(bb[:off] + g.tobytes())
# matching models: B points, A = R (B - cB) + d + cA with R the 90-deg turn in
# (x, z) of the rec (Y is thickness) and d = (-5, 0, -6), plus a +-1 pixel
# picking error so the fit is not degenerate
pts = [(120, 40, 130), (400, 50, 110), (390, 100, 400), (130, 110, 380),
       (256, 75, 256), (200, 60, 330), (330, 90, 180), (260, 30, 420)]
jit = [(1, 0, -1), (0, 1, 0), (-1, 0, 1), (0, -1, 1), (1, 1, 0), (-1, 0, 0), (0, 0, -1), (1, -1, 1)]
with open('gb.matpts', 'w') as fb, open('ga.matpts', 'w') as fa:
    for (x, y, z), (jx, jy, jz) in zip(pts, jit):
        fb.write('1 1 %d %d %d\n' % (x, y, z))
        fa.write('1 1 %d %d %d\n' % (-(z - 256) - 5 + 256 + jx, y + jy, (x - 256) - 6 + 256 + jz))
PY
    $REF/flib/image/newstack gb_raw.ali gb.ali >> dualsetup.log
    for x in a b; do
      $REF/imodutil/point2model -scat -volume 512,150,512 g$x.matpts g$x.matmod >> dualsetup.log
      ln -sfn g$x.ali g$x.st
    done
    rm -f a2.rec b2rot.rec gb_reproj.mrc gb_raw.ali )
}

# ------------------------------------------------------------------- steps
# S NAME PROG TOL "INPUTS" "OUTPUTS" [ARGS...]  <<stdin
#   TOL = exact   must be byte-identical (MRC label stamps/uninitialised
#                 slots and model-name residue masked)
#         lapack  derived from tiltalign's LAPACK solve (dspsv; faer stands in
#                 for it in imod-rs): byte-identical, or within the bound below
# Stdin is the step's parameter block, exactly as the .com would pass it.
NSTEP=0
S() {
  local n=$1; NAMES[$NSTEP]=$1; PROGS[$NSTEP]=$2; TOLS[$NSTEP]=$3
  INS[$NSTEP]=$(echo $4); OUTS[$NSTEP]=$(echo $5); shift 5
  ARGS[$NSTEP]="$*"
  cat > $B/steps/$n.in
  NSTEP=$((NSTEP + 1))
}
FILT="FilterSigma1	0.03
FilterRadius2	0.25
FilterSigma2	0.05"

# Coarse binning to 1024 x 1024 so the whole pipeline runs in minutes.
S 01_bin4 newstack exact "TS_01.mrc" "ts.st" -StandardInput <<EOF
InputFile	TS_01.mrc
OutputFile	ts.st
BinByFactor	4
EOF
S 02_stat_st clip exact "ts.st" "" stats ts.st </dev/null
# xcorr.com
S 03_xcorr tiltxcorr exact "ts.st ts.rawtlt" "ts.prexf" -StandardInput <<EOF
InputFile	ts.st
OutputFile	ts.prexf
TiltFile	ts.rawtlt
RotationAngle	-90.
$FILT
EOF
# prenewst.com: xftoxg (interactive), then newstack
S 04_xftoxg xftoxg exact "ts.prexf" "ts.prexg" <<EOF
0
ts.prexf
ts.prexg
EOF
S 05_prenewst newstack exact "ts.st ts.prexg" "ts.preali" -StandardInput <<EOF
InputFile	ts.st
OutputFile	ts.preali
TransformFile	ts.prexg
ModeToOutput	0
FloatDensities	2
BinByFactor	1
ImagesAreBinned	1
EOF
# xcorr_pt.com (fiducialless patch tracking).  imodchopconts is not in the
# command table; tiltxcorr's own LengthAndOverlap breaks the contours instead.
S 06_xcorr_pt tiltxcorr exact "ts.preali ts.prexg ts.rawtlt" "ts.fid" -StandardInput <<EOF
InputFile	ts.preali
OutputFile	ts.fid
PrealignmentTransformFile	ts.prexg
ImagesAreBinned	1
TiltFile	ts.rawtlt
RotationAngle	-90.
$FILT
BordersInXandY	51,51
IterateCorrelations	1
SizeOfPatchesXandY	160,160
OverlapOfPatchesXandY	0.33,0.33
LengthAndOverlap	16,4
EOF
# align.com (patch model; surfaces not assigned, so SurfacesToAnalyze 1)
S 07_align tiltalign lapack "ts.fid ts.preali ts.rawtlt" \
  "ts.3dmod ts.resid tsfid.xyz ts.tlt ts.xtilt ts.tltxf ts_nogaps.fid" -StandardInput <<EOF
ModelFile	ts.fid
ImageFile	ts.preali
ImagesAreBinned	1
OutputModelFile	ts.3dmod
OutputResidualFile	ts.resid
OutputFidXYZFile	tsfid.xyz
OutputTiltFile	ts.tlt
OutputXAxisTiltFile	ts.xtilt
OutputTransformFile	ts.tltxf
OutputFilledInModel	ts_nogaps.fid
RotationAngle	-90.0
TiltFile	ts.rawtlt
AngleOffset	0.
RotOption	1
RotDefaultGrouping	5
TiltOption	5
TiltDefaultGrouping	5
MagReferenceView	1
MagOption	1
MagDefaultGrouping	4
XStretchOption	0
SkewOption	0
XStretchDefaultGrouping	7
SkewDefaultGrouping	11
BeamTiltOption	0
XTiltOption	0
XTiltDefaultGrouping	2000
ResidualReportCriterion	3.0
SurfacesToAnalyze	1
MetroFactor	.25
MaximumCycles	1000
KFactorScaling	1.
NoSeparateTiltGroups	1
AxisZShift	0.
ShiftZFromOriginal	1
LocalAlignments	0
EOF
# align.com tail: combine with the prealignment
S 08_xfproduct xfproduct lapack "ts.prexg ts.tltxf" "ts.xf" -StandardInput <<EOF
InputFile1	ts.prexg
InputFile2	ts.tltxf
OutputFile	ts.xf
EOF
# newst.com (from ccdnewst.com)
S 09_newst newstack lapack "ts.st ts.xf" "ts.ali" -StandardInput <<EOF
InputFile	ts.st
OutputFile	ts.ali
TransformFile	ts.xf
TaperAtFill	1,0
AdjustOrigin
SizeToOutputInXandY	1024,1024
OffsetsInXandY	0,0
ImagesAreBinned	1
EOF
S 10_stat_ali clip lapack "ts.ali" "" stats ts.ali </dev/null
# tilt.com, float input -> MODE 2 (copytomocoms); LOG removed (cryo data)
TILTC="InputProjections	ts.ali
IMAGEBINNED	1
TILTFILE	ts.tlt
XTILTFILE	ts.xtilt
XAXISTILT	0.
PERPENDICULAR
FULLIMAGE	1024 1024
SUBSETSTART	0 0
AdjustOrigin	1"
S 11_tilt tilt lapack "ts.ali ts.tlt ts.xtilt" "ts.rec" -StandardInput <<EOF
$TILTC
OutputFile	ts.rec
THICKNESS	300
RADIAL	.35 .035
FalloffIsTrueSigma	1
SCALE	0 500
MODE	2
EOF
S 12_tilt_m1 tilt lapack "ts.ali ts.tlt ts.xtilt" "ts_m1.rec" -StandardInput <<EOF
$TILTC
OutputFile	ts_m1.rec
THICKNESS	150
RADIAL	.35 .035
FalloffIsTrueSigma	1
SCALE	0 500
MODE	1
EOF
S 13_tilt_defrad tilt lapack "ts.ali ts.tlt ts.xtilt" "ts_defrad.rec" -StandardInput <<EOF
$TILTC
OutputFile	ts_defrad.rec
THICKNESS	200
MODE	2
EOF
S 14_tilt_log tilt lapack "ts.ali ts.tlt ts.xtilt" "ts_log.rec" -StandardInput <<EOF
$TILTC
OutputFile	ts_log.rec
THICKNESS	200
RADIAL	.35 .035
FalloffIsTrueSigma	1
LOG	5
MODE	2
EOF
S 15_tilt_reproj tilt lapack "ts.ali ts.tlt ts.xtilt ts.rec" "ts_reproj.mrc" -StandardInput <<EOF
$TILTC
OutputFile	ts_reproj.mrc
THICKNESS	300
RecFileToReproject	ts.rec
EOF
S 16_tilt_repslc tilt lapack "ts.ali ts.tlt ts.xtilt" "ts_repslc.mrc" -StandardInput <<EOF
$TILTC
OutputFile	ts_repslc.mrc
THICKNESS	300
RADIAL	.35 .035
FalloffIsTrueSigma	1
REPROJECT	-30,0,30
EOF
S 17_stat_rec clip lapack "ts.rec" "" stats ts.rec </dev/null
# batchruntomo's runFindSection (positioning model form)
S 18_findsec findsection lapack "ts.rec" "ts_pitch.mod" -StandardInput <<EOF
TomogramFile	ts.rec
NumberOfDefaultScales	2
SizeOfBoxesInXYZ	16,1,16
TomoPitchModel	ts_pitch.mod
NumberOfSamples	5
EOF
S 19_trimvol trimvol lapack "ts.rec" "ts_trim.rec" -rx ts.rec ts_trim.rec </dev/null
S 20_trimvol_sz trimvol lapack "ts.rec" "ts_trimb.rec" -f -rx -sz 120,180 ts.rec ts_trimb.rec </dev/null
S 21_binvol binvol lapack "ts.rec" "ts_bin2.rec" -binning 2 ts.rec ts_bin2.rec </dev/null
S 22_mrc2tif mrc2tif lapack "ts_trimb.rec" "slice.tif" -s -z 150,150 ts_trimb.rec slice.tif </dev/null

# ===================== single-axis additions (2026-09-26) =====================
# Setup scripts: generate the data set's command files on each side and
# compare them.  -style 0 keeps the descriptive names (ts.ali, ts.rec) that the
# steps above use.
CTCCOMS="align.com ctfcorrection.com ctfplotter.com eraser.com mtffilter.com newst.com
  prenewst.com sample.com tilt.com tomopitch.com track.com xcorr.com"
S 23_ctc copytomocoms exact "ts.st ts.rawtlt" \
  "$CTCCOMS ts.seed $(for c in $CTCCOMS; do printf 'origcoms/%s ' $c; done)" \
  -name ts -style 0 -stackext st -pixel 0.4716 -gold 10 -rotation -90 -userawtlt \
  -voltage 300 -Cs 2.7 -defocus 3000 </dev/null
S 24_mkc_xcorrpt makecomfile exact "xcorr.com ts.st ts.preali" "xcorr_pt.com" \
  -root ts -style 0 -stackext st -binning 1 -input xcorr.com -output xcorr_pt.com </dev/null
S 25_mkc_tilt3df makecomfile exact "tilt.com ts.st" "tilt_3dfind.com" \
  -root ts -style 0 -stackext st -binning 2 -thickness 150 -input tilt.com -output tilt_3dfind.com </dev/null
S 26_mkc_fb3d makecomfile exact "ts.st ts.tlt" "findbeads3d.com" \
  -root ts -style 0 -stackext st -binning 2 -bead 21.2 -output findbeads3d.com </dev/null
S 27_mkc_reproj makecomfile exact "tilt.com ts.st" "tilt_3dfind_reproject.com" \
  -root ts -style 0 -stackext st -input tilt.com -output tilt_3dfind_reproject.com </dev/null
S 28_mkc_golder makecomfile exact "ts.st" "golderaser.com" \
  -root ts -style 0 -stackext st -bead 21.2 -output golderaser.com </dev/null
S 29_mkc_newst3df makecomfile exact "newst.com ts.st" "newst_3dfind.com" \
  -root ts -style 0 -stackext st -binning 2 -input newst.com -output newst_3dfind.com </dev/null
S 30_mkc_sirt makecomfile exact "ts.st" "sirtsetup.com" -root ts -style 0 -stackext st -output sirtsetup.com </dev/null
# eraser.com as copytomocoms wrote it (X-ray removal on the raw stack)
S 31_eraser comrun exact "eraser.com ts.st" "ts_fixed.st ts_peak.mod" eraser.com </dev/null
# The binned stack has no peaks at the template criteria: the same step with
# lowered criteria, so the replacement code actually runs (576 contours).
S 32_eraser_sens ccderaser exact "ts.st" "ts_fix5.st ts_peak5.mod" -StandardInput <<EOF
InputFile	ts.st
OutputFile	ts_fix5.st
FindPeaks
PeakCriterion	5
DiffCriterion	5
GrowCriterion	3.
EdgeExclusionWidth	4
PointModel	ts_peak5.mod
MaximumRadius	4.2
AnnulusWidth	2.0
XYScanSize	100
ScanCriterion	3.
BorderSize	2
PolynomialOrder	2
EOF
# The template criteria on the unbinned 4096^2 data, TrialMode (no output
# stack; the point model and the would-be min/max are the result)
S 33_eraser_raw ccderaser exact "TS_01.mrc" "raw_peak.mod" -StandardInput <<EOF
InputFile	TS_01.mrc
FindPeaks
PeakCriterion	10.
DiffCriterion	8.
BigDiffCriterion	19.
GiantCriterion	12.
ExtraLargeRadius	8.
GrowCriterion	4.
EdgeExclusionWidth	4
PointModel	raw_peak.mod
MaximumRadius	4.2
AnnulusWidth	2.0
XYScanSize	100
ScanCriterion	3.
BorderSize	2
PolynomialOrder	2
TrialMode
EOF
# Gold erasing as batchruntomo chains it: findbeads3d on a binned-by-2
# reconstruction (binvol's ts_bin2.rec standing in for tilt_3dfind's output),
# reproject the bead model onto the aligned stack, erase with ccderaser.
S 34_fb3d comrun lapack "findbeads3d.com ts_3dfind.rec=ts_bin2.rec ts.tlt" "ts_3dfind.mod" \
  findbeads3d.com </dev/null
S 35_reproj_fid comrun lapack "tilt_3dfind_reproject.com ts.ali ts.tlt ts.xtilt ts_3dfind.mod" \
  "ts_erase.fid" tilt_3dfind_reproject.com </dev/null
S 36_golderaser comrun lapack "golderaser.com ts.ali ts_erase.fid" "ts_erase.ali" \
  golderaser.com </dev/null
# ctfcorrection.com with the ASSUMED constant 3 um defocus file
S 37_ctfphaseflip comrun lapack "ctfcorrection.com ts.ali ts.tlt ts.xf ts.defocus" \
  "ts_ctfcorr.ali" ctfcorrection.com </dev/null
# mtffilter.com as generated: it names ts.mrc.mdoc but has no TypeOfDoseFile, so
# mtffilter never reads it (mtffilter.cpp:389-397) - a plain 2D filter
S 38_mtffilter comrun lapack "mtffilter.com ts.ali" "ts_filt.ali" mtffilter.com </dev/null
# Dose weighting from the SYNTHETIC mdoc (PriorRecordDose, dose-symmetric)
S 39_mtf_dose mtffilter lapack "ts.ali ts.st.mdoc" "ts_dose.ali" -StandardInput <<EOF
InputFile	ts.ali
OutputFile	ts_dose.ali
TypeOfDoseFile	4
DoseWeightingFile	ts.st.mdoc
Voltage	300
PixelSize	0.4716
EOF
# tomopitch.com on the SYNTHETIC three-sample model; its log is the result
S 40_tomopitch comrun exact "tomopitch.com tomopitch.mod" "tomopitch.log" tomopitch.com </dev/null
S 41_xcorrstack xcorrstack exact "ts.preali ts.st" "ts_xc.mrc" \
  -stack ts.preali -single ts.st -output ts_xc.mrc -rad2 0.25 -sig1 0.03 -sig2 0.05 </dev/null
# TS_01 has no tilt angles in its header: both sides must fail the same way
S 42_exttilt_hdr extracttilts exact "TS_01.mrc" "ts_hdr.tlt" -input TS_01.mrc -output ts_hdr.tlt </dev/null
S 43_exttilt_mdoc extracttilts exact "ts.st ts.st.mdoc" "ts_mdoc.tlt" \
  -mdoc -input ts.st -output ts_mdoc.tlt </dev/null
S 44_exttilt_dose extracttilts exact "ts.st ts.st.mdoc" "ts_mdoc.dose" \
  -mdoc -aretomo -input ts.st -output ts_mdoc.dose </dev/null
# splittilt the generated tilt.com and run the chunks with processchunks
SPLITCOMS="$(for n in $(seq -w 1 20); do printf 'tilt-0%s.com ' $n; done)tilt-start.com tilt-finish.com"
S 45_splittilt splittilt lapack "tilt.com ts.ali ts.tlt ts.xtilt" "$SPLITCOMS tilt-bound.info" \
  -n 4 tilt.com </dev/null
S 46_processchunks processchunks lapack "tilt.com $SPLITCOMS tilt-bound.info ts.ali ts.tlt ts.xtilt" \
  "ts.rec" -g localhost,localhost,localhost,localhost tilt </dev/null

# ===================== dual-axis combine (SYNTHETIC B axis) =====================
# See dual_setup above.  Every command file comes from each side's own
# copytomocoms / setupcombine and runs through comrun, except where a step is
# one piece of volcombine.com run on its own (transcribed from it).
DCOMS="aligna.com alignb.com ctfcorrectiona.com ctfcorrectionb.com ctfplottera.com
  ctfplotterb.com erasera.com eraserb.com mtffiltera.com mtffilterb.com newsta.com newstb.com
  prenewsta.com prenewstb.com samplea.com sampleb.com tilta.com tiltb.com tomopitcha.com
  tomopitchb.com tracka.com trackb.com xcorra.com xcorrb.com"
S 50_d_ctc copytomocoms exact "ga.st gb.st ga.rawtlt gb.rawtlt" "$DCOMS ga.seed gb.seed" \
  -name g -dual -style 0 -stackext st -pixel 0.9432 -gold 10 -rotation 0 -brotation 0 \
  -userawtlt -buserawtlt -voltage 300 -Cs 2.7 -defocus 3000 \
  -one comparam.tilta.tilt.THICKNESS=150 -one comparam.tiltb.tilt.THICKNESS=150 </dev/null
S 51_d_tilta comrun exact "tilta.com ga.ali ga.tlt ga.xtilt" "ga.rec" tilta.com </dev/null
S 52_d_tiltb comrun exact "tiltb.com gb.ali gb.tlt gb.xtilt" "gb.rec" tiltb.com </dev/null
S 53_d_setup setupcombine exact "ga.rec gb.rec ga.st gb.st tilta.com tiltb.com ga.tlt gb.tlt" \
  "combine.com solvematch.com dualvolmatch.com matchvol1.com patchcorr.com matchorwarp.com
   warpvol.com matchvol2.com volcombine.com" \
  -name g -initial -patchsize S -zlimits 1,150 -stackext st </dev/null
S 54_d_dualvol comrun lapack "dualvolmatch.com ga.rec gb.rec ga.tlt gb.tlt ga.st gb.st ga.ali gb.ali tilta.com tiltb.com" \
  "solve.xf" dualvolmatch.com </dev/null
# solvematch from the SYNTHETIC matching models (an alternative to 54)
S 55_d_solvematch solvematch lapack "ga.rec gb.rec ga.matmod gb.matmod" "solve_mm.xf" -StandardInput <<EOF
XAxisTilts	0,0
AngleOffsetsToTilt	0,0
ZShiftsToTilt	0,0
SurfacesOrUseModels	-2
AMatchingModel	ga.matmod
BMatchingModel	gb.matmod
MatchingAtoB	0
ATomogramOrSizeXYZ	ga.rec
BTomogramOrSizeXYZ	gb.rec
OutputFile	solve_mm.xf
MaximumResidual	8.
LocalFitting	10
CenterShiftLimit	10.
EOF
S 56_d_matchvol1 comrun lapack "matchvol1.com gb.rec solve.xf" "gb.mat inverse.xf" matchvol1.com </dev/null
S 57_d_patchcorr comrun lapack "patchcorr.com ga.rec gb.mat gb.rec solve.xf" \
  "patch.out patch_vector_ccc.mod" patchcorr.com </dev/null
S 58_d_matchorwarp comrun lapack "matchorwarp.com ga.rec gb.rec gb.mat patch.out solve.xf inverse.xf" \
  "gb.mat refine.xf inverse.xf patch.resid" matchorwarp.com </dev/null
# volcombine.com, piece by piece (in-place steps get a copy of their input)
S 59_d_densmatch densmatch lapack "ga.rec +gb.mat" "gb.mat" <<EOF
ga.rec
gb.mat

EOF
S 60_d_filltomo filltomo lapack "+gb.mat ga.rec gb.rec inverse.xf" "gb.mat" -StandardInput <<EOF
FillTomogram	gb.mat
MatchedToTomogram	ga.rec
SourceTomogram	gb.rec
InverseTransformFile	inverse.xf
EOF
S 61_d_combinefft combinefft lapack "ga.rec gb.mat inverse.xf ga.tlt gb.tlt" "sum1.rec" -StandardInput <<EOF
AInputFFT	ga.rec
BInputFFT	gb.mat
OutputFFT	sum1.rec
XMinAndMax	0,511
YMinAndMax	0,149
ZMinAndMax	0,511
TaperPadsInXYZ	8,4,8
InverseTransformFile	inverse.xf
ATiltFile	ga.tlt
BTiltFile	gb.tlt
ReductionFraction	0
LowFromBothRadius	0
EOF
S 62_d_assemble assemblevol lapack "sum1.rec" "sum.rec" -StandardInput <<EOF
OutputFile sum.rec
StartEndToExtractInX 14,525
StartEndToExtractInY 5,154
StartEndToExtractInZ 14,525
InputFile sum1.rec
EOF
S 63_d_fillsum filltomo lapack "+sum.rec ga.rec gb.rec inverse.xf" "sum.rec" -StandardInput <<EOF
FillTomogram	sum.rec
MatchedToTomogram	ga.rec
SourceTomogram	gb.rec
InverseTransformFile	inverse.xf
EOF
# ... and the whole volcombine.com, as combine.com runs it
S 64_d_volcombine comrun lapack \
  "volcombine.com ga.rec gb.rec +gb.mat:58_d_matchorwarp inverse.xf:58_d_matchorwarp ga.tlt gb.tlt" \
  "sum.rec gb.mat" volcombine.com </dev/null
S 65_d_splitcomb splitcombine exact "volcombine.com" \
  "volcombine-start.com volcombine-001.com volcombine-finish.com" volcombine.com </dev/null
# Branches (after the main chain so their gb.mat does not shadow it):
# matchorwarp forced to warp (RefineLimit below refinematch's residual) runs
# findwarp + warpvol; autopatchfit re-runs patchcorr/matchorwarp itself.
S 66_d_mowwarp matchorwarp lapack \
  "ga.rec gb.rec gb.mat:56_d_matchvol1 patch.out solve.xf inverse.xf:56_d_matchvol1" \
  "gb.mat warp.xf patch.resid" -StandardInput <<EOF
InputVolume	gb.rec
OutputVolume	gb.mat
SizeXYZorVolume	ga.rec
RefineLimit	0.01
WarpLimits	0.2,0.27,0.35
ResidualFile	patch.resid
ClipPlaneBoxSize	600
EOF
S 67_d_autopatch autopatchfit lapack \
  "+patchcorr.com +matchorwarp.com ga.rec gb.rec solve.xf +gb.mat:56_d_matchvol1 +inverse.xf:56_d_matchvol1 ga.tlt gb.tlt" \
  "gb.mat refine.xf patch.out patch.resid patch_vector_ccc.mod" -final M </dev/null

for ((i = 0; i < NSTEP; i++)); do
  [[ " $RSCMDS comrun " == *" ${PROGS[$i]} "* ]] || echo "WARNING: ${PROGS[$i]} not in the Rust command table" >&2
done

# ---------------------------------------------------------------- the runner
# producer DIRROOT FILE STEPINDEX -> directory holding FILE for step STEPINDEX
producer() {
  local root=$1 f=$2 i=$3 j
  for ((j = i - 1; j >= 0; j--)); do
    [[ " ${OUTS[$j]} " == *" $f "* ]] && { echo $root/${NAMES[$j]}; return; }
  done
  echo $B/data
}

# An INS entry is  [+]LOCAL[=SOURCE][:STEP]
#   +       copy the file instead of linking it (the step rewrites it in place)
#   =SOURCE the producer's file is called SOURCE (linked in as LOCAL)
#   :STEP   take it from that step rather than the latest earlier producer
# insrc INROOT SPEC STEPINDEX -> "copy local path"
insrc() {
  local spec=$2 copy=0 step= src dir
  [[ $spec == +* ]] && { copy=1; spec=${spec#+}; }
  [[ $spec == *:* ]] && { step=${spec#*:}; spec=${spec%%:*}; }
  src=$spec; [[ $spec == *=* ]] && { src=${spec#*=}; spec=${spec%%=*}; }
  if [ -n "$step" ]; then dir=$1/$step; else dir=$(producer $1 $src $3); fi
  echo $copy $spec $dir/$src
}

# run SIDE DIR INPUTROOT STEPINDEX [OMP]  -> writes DIR/.time "wall user sys rss rc"
run() {
  local side=$1 d=$2 inroot=$3 i=$4 omp=$5 bin env f copy loc src
  rm -rf $d; mkdir -p $d
  for f in ${INS[$i]}; do
    read copy loc src <<< "$(insrc $inroot $f $i)"
    if [ $copy = 1 ]; then cp $src $d/$loc; else ln -s $src $d/$loc; fi
  done
  if [ $side = nat ]; then
    bin=$NTI/bin/${PROGS[$i]}
    env=(IMOD_DIR=$NTI PATH=$NTI/bin:$PATH LD_LIBRARY_PATH=$REF/buildlib)
  else
    bin=$RSI/bin/${PROGS[$i]}
    env=(IMOD_DIR=$RSI PATH=$RSI/bin:$PATH)
  fi
  [ -n "$omp" ] && env+=(OMP_NUM_THREADS=$omp)
  ( cd $d && /usr/bin/time -f "%e %U %S %M" -o .time \
      env AUTODOC_DIR=$REF/autodoc "${env[@]}" timeout 3600 $bin ${ARGS[$i]} \
      < $B/steps/${NAMES[$i]}.in 2> err.txt | cat > out.txt
    rc=${PIPESTATUS[0]}; t=$(tail -1 .time); echo "$t $rc" > .time )
}

cat > $B/compare.py <<'PY'
# compare.py TOL NATDIR RSDIR FILE...  -> one verdict line
# Verdicts: IDENT (byte-identical after masking), TOL (numeric difference
# within the stated LAPACK bound), DIFF, MISSING.
#
# LAPACK bound (CLAUDE.md: "state a bound and check it"):
#   text / model numbers : |n - r| <= 1e-4 * max(1, |n|)  per value
#   image voxels         : max|n - r| <= 1e-3 * (native max - native min),
#                          and NaN at the same voxels
#   byte/short images    : additionally a +-1 count is allowed (a rounding
#                          flip of a value within the float bound)
import os, re, struct, sys
import numpy as np
tol, nd, rd = sys.argv[1], sys.argv[2], sys.argv[3]
RTXT, RIMG = 1e-4, 1e-3
out = []
worst = 'IDENT'
rank = {'IDENT': 0, 'TOL': 1, 'DIFF': 2, 'MISSING': 3}

def bump(v):
    global worst
    if rank[v] > rank[worst]: worst = v

def ismrc(b):
    return len(b) >= 1024 and b[208:212] in (b'MAP ', b'MAP\x00')

def mask_mrc(x):
    # mrc_head_new leaves labels past nlabl uninitialised (BUGS.md 2); the
    # dd-Mmm-yy HH:MM:SS stamp differs by run time.
    x = bytearray(x)
    nlabl = struct.unpack('<i', bytes(x[220:224]))[0]
    if 0 <= nlabl <= 10:
        for i in range(10):
            s = 224 + 80 * i
            if i < nlabl: x[s + 56:s + 76] = b' ' * 20
            else: x[s:s + 80] = b' ' * 80
    return bytes(x)

def mrc_array(b):
    nx, ny, nz, mode = struct.unpack('<4i', b[:16])
    next_ = struct.unpack('<i', b[92:96])[0]
    dt = {0: np.uint8, 1: np.int16, 2: np.float32, 6: np.uint16}[mode]
    # mode 0 may be signed; treat by the header's imodFlags bit 0
    if mode == 0 and struct.unpack('<i', b[152:156])[0] & 1: dt = np.int8
    return np.frombuffer(b, dt, nx * ny * nz, 1024 + next_), mode

def mask_model(b):
    # Imod.name[128] at offset 8: bytes past the NUL are heap residue
    x = bytearray(b)
    if x[:4] == b'IMOD':
        nul = x.find(b'\0', 8, 136)
        if nul >= 0: x[nul:136] = b'\0' * (136 - nul)
        # imodSetRefImage/putimageref malloc the IrefImage and never set
        # oscale or orot (imodel.c:1774, imodel_fwrap.c:2013), so the MINX
        # chunk's first and third vectors are heap residue (CLAUDE.md).
        p = x.find(b'MINX\0\0\0\x48')
        if p >= 0:
            x[p + 8:p + 20] = b'\0' * 12
            x[p + 32:p + 44] = b'\0' * 12
    return bytes(x)

NUM = re.compile(rb'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|NaN|nan|-?[Ii]nf')
TMPPID = re.compile(rb'\.tmp\.\d+')
STAMP = re.compile(rb'\d\d-[A-Z][a-z][a-z]-\d\d +\d\d:\d\d:\d\d')
PID = re.compile(rb'(Shell|Python) PID: *\d+')
def text_cmp(a, b):
    # label/banner date stamps are run time, not data
    a, b = STAMP.sub(b'<stamp>', a), STAMP.sub(b'<stamp>', b)
    a, b = TMPPID.sub(b'.tmp.<pid>', a), TMPPID.sub(b'.tmp.<pid>', b)
    a, b = PID.sub(b'PID', a), PID.sub(b'PID', b)
    if a == b: return 'IDENT', 'date stamp only'
    ta, tb = NUM.split(a), NUM.split(b)
    na, nb = NUM.findall(a), NUM.findall(b)
    if len(na) != len(nb) or [s.split() for s in ta] != [s.split() for s in tb]:
        return 'DIFF', 'structure differs'
    m, n = 0.0, 0
    for p, q in zip(na, nb):
        if p == q: continue
        n += 1
        fp, fq = float(p), float(q)
        m = max(m, abs(fp - fq) / max(1.0, abs(fp)))
    if n == 0: return 'IDENT', 'whitespace only'
    v = 'TOL' if (tol == 'lapack' and m <= RTXT) else 'DIFF'
    return v, '%d of %d numbers differ, max rel %.3g' % (n, len(na), m)

def model_text(path):
    import subprocess
    ref = os.environ.get('REF', '/tmp/imod-reference-build')
    r = subprocess.run([ref + '/imodutil/imodinfo', '-a', path], capture_output=True,
                       env=dict(os.environ, LD_LIBRARY_PATH=ref + '/buildlib'))
    return r.stdout

for f in sys.argv[4:]:
    pn, pr = os.path.join(nd, f), os.path.join(rd, f)
    if not os.path.exists(pn) and not os.path.exists(pr):
        out.append('%s:neither-wrote' % f); continue
    if not os.path.exists(pn):
        out.append('%s:native-missing' % f); bump('MISSING'); continue
    if not os.path.exists(pr):
        out.append('%s:MISSING' % f); bump('MISSING'); continue
    a, b = open(pn, 'rb').read(), open(pr, 'rb').read()
    if ismrc(a) and ismrc(b):
        ha, hb = mask_mrc(a[:1024]), mask_mrc(b[:1024])
        if len(a) != len(b):
            out.append('%s:DIFF(size %d vs %d)' % (f, len(a), len(b))); bump('DIFF'); continue
        if ha == hb and a[1024:] == b[1024:]:
            out.append('%s:IDENT' % f); continue
        hd = sum(1 for i, j in zip(ha, hb) if i != j)
        xa, mode = mrc_array(a); xb, _ = mrc_array(b)
        fa, fb = xa.astype(np.float64), xb.astype(np.float64)
        nana, nanb = np.isnan(fa), np.isnan(fb)
        ok = np.array_equal(nana, nanb)
        d = np.abs(np.where(nana, 0, fa) - np.where(nanb, 0, fb))
        rng = float(np.nanmax(fa) - np.nanmin(fa)) or 1.0
        nd_ = int((d > 0).sum())
        lim = RIMG * rng
        if mode in (0, 1, 6): lim = max(lim, 1.0)
        v = 'TOL' if (tol == 'lapack' and ok and d.max() <= lim) else 'DIFF'
        # the header min/max/mean follow the data; for a TOL image the header
        # difference is judged by the same bound through the data
        if v == 'TOL' or nd_ == 0:
            if nd_ == 0 and hd:
                v = 'DIFF'
        out.append('%s:%s(hdr %dB, %d of %d voxels, max|d| %.4g = %.3g of range, mean|d| %.3g)'
                   % (f, v, hd, nd_, d.size, d.max(), d.max() / rng, d.mean()))
        bump(v); continue
    if a[:2] in (b'II', b'MM'):
        DT = re.compile(rb'\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}')
        a2, b2 = DT.sub(b'?' * 19, a), DT.sub(b'?' * 19, b)
        if a2 == b2: out.append('%s:IDENT' % f); continue
        if len(a2) == len(b2):
            n = sum(1 for i, j in zip(a2, b2) if i != j)
            # uncompressed byte TIFF: pixels differing by <= 1 count are TOL
            dd = np.abs(np.frombuffer(a2, np.uint8).astype(int) - np.frombuffer(b2, np.uint8))
            v = 'TOL' if (tol == 'lapack' and dd.max() <= 1) else 'DIFF'
            out.append('%s:%s(%d bytes, max %d)' % (f, v, n, dd.max())); bump(v); continue
        out.append('%s:DIFF(size)' % f); bump('DIFF'); continue
    if a[:4] == b'IMOD':
        ma, mb = mask_model(a), mask_model(b)
        if ma == mb: out.append('%s:IDENT' % f); continue
        v, why = text_cmp(model_text(pn), model_text(pr))
        if v == 'IDENT': v = 'DIFF'; why = 'binary differs, ascii same'
        out.append('%s:%s(%s)' % (f, v, why)); bump(v); continue
    if a == b: out.append('%s:IDENT' % f); continue
    v, why = text_cmp(a, b)
    if v == 'IDENT': v = 'DIFF'
    out.append('%s:%s(%s)' % (f, v, why)); bump(v)

print(worst + '\t' + ' '.join(out))
PY

t4() { cat $1/.time; }   # "wall user sys rss rc"
cpu() { awk '{printf "%.2f", $2 + $3}' $1/.time; }

# stdout: data for clip stats / findsection; informational elsewhere
outcmp() {
  if cmp -s $1/out.txt $2/out.txt; then echo same
  else python3 $B/compare.py lapack $1 $2 out.txt | cut -f2 | sed 's/^out.txt://'; fi
}

report_line() { # phase i natdir rsdir
  local i=$4 v
  v=$(REF=$REF python3 $B/compare.py ${TOLS[$i]} $2 $3 ${OUTS[$i]})
  read nw nu ns nm nrc < $2/.time; read rw ru rs rm rrc < $3/.time
  [ "$nrc" != "$rrc" ] && v="RC($nrc/$rrc) $v"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "${NAMES[$i]}" "${PROGS[$i]}" \
    "$nw" "$(cpu $2)" "$rw" "$(cpu $3)" "$nrc/$rrc" "${v%%	*}" "${v#*	}" \
    "stdout:$(outcmp $2 $3)" | tee -a $B/results.tsv
}

sel() { [ -z "$ONLY" ] || [[ ${NAMES[$1]} =~ $ONLY ]]; }

[ -z "$ONLY" ] && : > $B/results.tsv
for ((i = 0; i < NSTEP; i++)); do
  if sel $i && [[ ${NAMES[$i]} == *_d_* ]]; then
    [ -f $B/data/gb.ali ] || dual_setup
    break
  fi
done
for phase in $PHASES; do
  for ((i = 0; i < NSTEP; i++)); do
    sel $i || continue
    case $phase in
      pipe)
        run nat $B/pipe/nat/${NAMES[$i]} $B/pipe/nat $i
        run rs  $B/pipe/rs/${NAMES[$i]}  $B/pipe/rs  $i
        report_line pipe $B/pipe/nat/${NAMES[$i]} $B/pipe/rs/${NAMES[$i]} $i ;;
      cross)
        fromsetup=1
        for f in ${INS[$i]}; do
          read copy loc src <<< "$(insrc X $f $i)"; [ ${src%/*} = $B/data ] || fromsetup=0
        done
        [ $fromsetup = 1 ] && continue
        run rs $B/cross/${NAMES[$i]} $B/pipe/nat $i
        report_line cross $B/pipe/nat/${NAMES[$i]} $B/cross/${NAMES[$i]} $i ;;
      omp1)
        run nat $B/omp1/nat/${NAMES[$i]} $B/omp1/nat $i 1
        run rs  $B/omp1/rs/${NAMES[$i]}  $B/omp1/rs  $i 1
        report_line omp1 $B/omp1/nat/${NAMES[$i]} $B/omp1/rs/${NAMES[$i]} $i
        # thread-count independence of each side
        for s in nat rs; do
          v=$(REF=$REF python3 $B/compare.py exact $B/pipe/$s/${NAMES[$i]} $B/omp1/$s/${NAMES[$i]} ${OUTS[$i]} | cut -f1)
          echo "  omp1-vs-default $s ${NAMES[$i]}: $v" | tee -a $B/results.tsv
        done ;;
    esac
  done
done
echo "results: $B/results.tsv" >&2
