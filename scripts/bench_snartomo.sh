#!/bin/bash
# SNARTomo's IMOD invocations, native IMOD vs imod-rs: speed, peak RSS and
# output parity.  Results and the command inventory are in BENCHMARK.md.
#
# Every case below is a command line SNARTomo (https://github.com/rubenlab/
# snartomo) runs, copied from its source with the same options; the comment
# above each case names the SNARTomo function and file:line.  Inputs are made
# from real data by NATIVE IMOD (so neither side reads its own output), and
# each case is fed native inputs ("cross-fed"), so a difference belongs to that
# command alone.
#
#   MODE     setup | bench | parity | all   (default all; bench also checks parity)
#   THREADS  thread settings to run, "default 1" by default; "1" means
#            OMP_NUM_THREADS=1 on both sides (imod-rs honours it too)
#   REPS     runs per side per setting (default 3; the median is reported);
#            cases under 0.2 s are re-timed with `perf stat -r SHORTREPS` (20)
#   ONLY     extended regex on case names; SKIP: regex of cases to leave out
#   B        scratch root (default /big/henriksson/realbench/snarbench, ~12 GB)
#   RSBIN    imod-rs binary (copied before use)
#   REF      native reference build
#   SELFCHECK  regex of cases where each side is also compared with ITSELF
#            (previous rep vs last rep; needs REPS >= 2), default '^batchruntomo'
#            -- separates run-to-run variation (native beadtrack, RAPTOR's
#            clock seed) from translation differences
#   PARITYREPS  runs per side in MODE=parity (default 1; 2 for a self-check)
#   NATIVE_SUBST  "prog=/path/to/binary ..." -- programs the native side runs
#            from elsewhere instead of the reference build: native sources
#            rebuilt with a BUGS.md fix, to attribute a parity difference to
#            that fix (see BENCHMARK.md "Attributing the differences").  The
#            row's note then says "nat+subst"; it is not stock native IMOD.
#   RAPTOR_SEED    if set, RAPTOR runs with `-seed N` on both sides (through a
#            RAPTOR_BIN wrapper; batchruntomo/runraptor have no seed option, so
#            this is a parity diagnostic, not a SNARTomo command line)
#
# Output: one table row per case and thread setting --
#   wall (median), CPU = perf task-clock of the whole process tree (median), max RSS (max over runs), ratios
#   native/imod-rs (>1.00x = imod-rs faster / less CPU), RSS ratio imod-rs/native
#   (>1.00x = imod-rs uses more), PARITY (compare verdict for every output file,
#   plus stdout for the cases where SNARTomo parses stdout).
# Raw per-run numbers: $B/results/<stamp>/runs.tsv; table: table.txt.
export LC_ALL=C

REPO=$(cd "$(dirname "$0")/.." && pwd)
REF=${REF:-/tmp/imod-reference-build}
RSBIN=${RSBIN:-$REPO/target/release/imod}
B=${B:-/big/henriksson/realbench/snarbench}
MODE=${MODE:-all}
THREADS=${THREADS:-default 1}
REPS=${REPS:-3}
ONLY=${ONLY:-}
SKIP=${SKIP:-}
TS=${TS:-/husky/otherdataset/teresa/EM/4johan/TS_01.mrc}
MOVIES=${MOVIES:-/big/henriksson/realbench/data}
E2E=${E2E:-/big/henriksson/realbench/e2e-ts01}

[ -d $REF/flib ] || ln -sfn /big/henriksson/imod-reference-build /tmp/imod-reference-build
[ -x $RSBIN ] || { echo "build first: cargo build --release --bin imod" >&2; exit 1; }
D=$B/data
mkdir -p $D $B/run $B/results

# ------------------------------------------------------------------ installs
# Same layout as scripts/e2e_ts01.sh: each side is an IMOD_DIR with bin/ (one
# link per command), com/ and, for native, pylib/ for the Python scripts
# (batchruntomo, submfg, trimvol).  A case runs with PATH = that bin only, so
# a script's child commands stay on its own side.
RSI=$B/rsimod; rm -rf $RSI; mkdir -p $RSI/bin
cp $RSBIN $RSI/imod; RSBIN=$RSI/imod
RSHASH=$(sha256sum < $RSBIN | cut -c1-16)
RSCMDS=$($RSBIN 2>&1 | sed -n '/^Commands:/,$p' | tail -n +2 | awk '{print $1}' | tr '\n' ' ')
for c in $RSCMDS; do ln -s $RSBIN $RSI/bin/$c; done
NTI=$B/natimod; rm -rf $NTI; mkdir -p $NTI/bin
ln -s $REF/pysrc $NTI/pylib
for d in flib/image flib/model flib/tilt flib/tiltalign flib/beadtrack flib/blend \
         flib/distort imodutil clip mrc qttools/mrc2tif qttools/processchunks pysrc \
         scripts raptor; do
  for f in $REF/$d/*; do
    n=$(basename $f)
    [ -f $f ] && [ -x $f ] && [[ $n != *.* ]] && [ ! -e $NTI/bin/$n ] && ln -s $f $NTI/bin/$n
  done
done
ln -s $REF/com $NTI/com; ln -s $REF/com $RSI/com
# RAPTOR (batchruntomo trackingMethod 2 -> runraptor -> $IMOD_DIR/bin/RAPTOR,
# which runs MarkersCorrespond from the same directory).
for sub in $NATIVE_SUBST; do
  [ -x "${sub#*=}" ] || { echo "NATIVE_SUBST: ${sub#*=} not executable" >&2; exit 1; }
  ln -sfn "${sub#*=}" $NTI/bin/${sub%%=*}; echo "NOTE: native side runs ${sub%%=*} from ${sub#*=}" >&2
done
if [ -n "$RAPTOR_SEED" ]; then
  for I in $NTI $RSI; do
    mkdir -p $I/rapseed; ln -sfn $(readlink $I/bin/MarkersCorrespond || echo $I/bin/MarkersCorrespond) $I/rapseed/MarkersCorrespond
    printf '#!/bin/sh\nexec %s "$@" -seed %s\n' $I/bin/RAPTOR $RAPTOR_SEED > $I/rapseed/RAPTOR; chmod +x $I/rapseed/RAPTOR
  done
  echo "NOTE: RAPTOR runs with -seed $RAPTOR_SEED on both sides" >&2
fi
# vmstopy (the .com -> Python converter processchunks/submfg use) is an
# external process boundary in the translation too; the Rust side gets the
# native script and the pylib it imports.  Nothing else on that side is native.
[ -e $RSI/bin/vmstopy ] || ln -s $REF/pysrc/vmstopy $RSI/bin/vmstopy; ln -s $REF/pysrc $RSI/pylib
# batchruntomo sets a data set up by running `etomo --fromBRT` (Java eTomo,
# an external boundary on both sides; the reference build's jar).
ln -s $REF/Etomo/jar_dir/etomo.jar $NTI/bin/etomo.jar; ln -s $REF/Etomo/jar_dir/etomo.jar $RSI/bin/etomo.jar
# IMOD's Python scripts start with `#!/usr/bin/env python`.
ln -s /usr/bin/python3 $NTI/bin/python
# Java eTomo runs IMOD's Python scripts as `python -u $IMOD_DIR/bin/<script>`
# (CopyTomoComs.java:294).  In an imod-rs install those names are the imod-rs
# binary, so the Rust side's `python` runs an ELF "script" directly and hands
# anything else to Python.
cat > $RSI/bin/python <<'EOF'
#!/bin/bash
a=("$@"); i=0
while [ $i -lt ${#a[@]} ] && [[ ${a[$i]} == -* ]]; do i=$((i + 1)); done
if [ $i -lt ${#a[@]} ] && [ "$(head -c4 "${a[$i]}" 2>/dev/null)" = $'\x7fELF' ]; then
  exec "${a[@]:$i}"
fi
exec /usr/bin/python3 "$@"
EOF
chmod +x $RSI/bin/python
# Helpers eTomo's setup calls that are not in the imod-rs command table run
# natively on the Rust side too (none of them is a SNARTomo command;
# archiveorig is a native Python script whose own children are imod-rs).
for c in montagesize archiveorig subimage; do
  # A wrapper, not a link: the native program needs the reference build's
  # libraries, and the imod-rs side runs without LD_LIBRARY_PATH (an imod-rs
  # install needs none; setting it would cost every imod-rs start ~300 failed
  # library probes that native's own install does not pay).
  [ -e $RSI/bin/$c ] || { printf '#!/bin/sh\nLD_LIBRARY_PATH=%s exec %s "$@"\n' $REF/buildlib $(readlink $NTI/bin/$c) > $RSI/bin/$c
                          chmod +x $RSI/bin/$c; echo "NOTE: imod-rs side uses native $c" >&2; }
done
# Commands SNARTomo needs that the translation does not have fall back to
# native on the Rust side, and are reported (none expected).
for c in header newstack alterheader binvol mrc2tif clip tif2mrc trimvol convertmod \
         imodinfo wmod2imod imodjoin submfg batchruntomo \
         autofidseed imodfindbeads imodmop clipmodel point2model sortbeadsurfs pickbestseed \
         beadtrack restrictalign imodchopconts runraptor RAPTOR MarkersCorrespond xfmodel; do
  [ -e $RSI/bin/$c ] || echo "WARNING: imod-rs has no '$c'" >&2
done

side_env() { # side -> env assignments
  local rb=""
  if [ $1 = rs ]; then
    [ -n "$RAPTOR_SEED" ] && rb=" RAPTOR_BIN=$RSI/rapseed"
    echo "PATH=$RSI/bin:/usr/bin:/bin IMOD_DIR=$RSI AUTODOC_DIR=$REF/autodoc$rb"
  else
    [ -n "$RAPTOR_SEED" ] && rb=" RAPTOR_BIN=$NTI/rapseed"
    echo "PATH=$NTI/bin:/usr/bin:/bin IMOD_DIR=$NTI AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib$rb"
  fi
}
nat() { env -u LD_LIBRARY_PATH $(side_env nat) "$@"; }

# ---------------------------------------------------------------- input data
# TS_01: 4096 x 4096 x 35 float, 1.179 A, real detector data (read only).  No
# tilt angles in the header: -51..+51 step 3 is ASSUMED (as in E2E.md).
# SNARTomo receives one MotionCor2 micrograph per tilt, in ACQUISITION order,
# plus one CTFFIND power spectrum per tilt, and restacks both in TILT order.
# We split TS_01 into 35 single-section MRC files named by a dose-symmetric
# acquisition index (0, +3, -3, +6, -6, ...), and make each view's power
# spectrum with native `clip spectrum` (512 x 512 float, CTFFIND's default
# box; CTFFIND itself is not IMOD).
setup_data() {
  cd $D || exit 1
  ln -sfn $TS TS_01.mrc
  ln -sfn $MOVIES/movie.eer movie.eer; ln -sfn $MOVIES/movie.tif movie.tif
  mkdir -p mics ctfs
  python3 - <<'PY'
angles = [-51 + 3 * i for i in range(35)]
# dose-symmetric acquisition order starting at 0 deg
order = sorted(range(35), key=lambda i: (abs(angles[i]), angles[i] < 0))
acq = {sec: k for k, sec in enumerate(order)}
open('secmap.txt', 'w').write(''.join('%d %d\n' % (s, acq[s]) for s in range(35)))
# SNARTomo's lists (snartomo-shared.bash:2977-3010): count, then name + "/" per tilt
with open('TS_01_mcorr.txt', 'w') as m, open('TS_01_ctfs.txt', 'w') as c:
    m.write('35\n'); c.write('35\n')
    for s in range(35):
        m.write('mics/TS_01_%03d.mrc\n/\n' % acq[s])
        c.write('ctfs/TS_01_%03d_ctf.mrc\n/\n' % acq[s])
open('TS_01_newstack.rawtlt', 'w').write(''.join('%.2f\n' % a for a in angles))
PY
  while read s a; do
    f=mics/TS_01_$(printf %03d $a).mrc
    [ -f $f ] || nat newstack -secs $s TS_01.mrc $f > /dev/null
    g=ctfs/TS_01_$(printf %03d $a)_ctf.mrc
    [ -f $g ] || nat clip spectrum -ox 512 -l 0 -m 2 $f $g > /dev/null
  done < secmap.txt
  # Gain reference (SNARTomo converts a TIFF gain with tif2mrc): a 32-bit
  # float TIFF made from the real movie -- the 45-frame average normalised to
  # mean 1 -- written by native mrc2tif.
  if [ ! -f gain.tif ]; then
    nat clip average -2d -m 2 movie.tif avg.tmp.mrc > /dev/null
    m=$(nat header avg.tmp.mrc | awk '/Mean density/{print $NF}')
    nat newstack -multadd $(awk -v m=$m 'BEGIN{printf "%.6g", 1/m}'),0 avg.tmp.mrc gain.tmp.mrc > /dev/null
    nat mrc2tif gain.tmp.mrc gain.tif > /dev/null
    rm -f avg.tmp.mrc gain.tmp.mrc
  fi
  # Native products the later SNARTomo steps consume.
  [ -f TS_01_newstack.mrc ] || { nat newstack -filei TS_01_mcorr.txt -ou TS_01_newstack.mrc >/dev/null;
                                 nat alterheader -PixelSize 1.179,1.179,1.179 TS_01_newstack.mrc >/dev/null; }
  [ -f TS_01_ctfstack.mrcs ] || nat newstack -filei TS_01_ctfs.txt -ou TS_01_ctfstack.mrcs >/dev/null
  [ -f TS_01_bin.mrcs ] || nat binvol -z 1 -x 16 -y 16 TS_01_newstack.mrc TS_01_bin.mrcs >/dev/null
  [ -f TS_01_ctfstack_center.mrcs ] || \
    nat clip resize -2d -ox 136 -oy 136 TS_01_ctfstack.mrcs TS_01_ctfstack_center.mrcs >/dev/null
  write_directive > snartomo.adoc
  for m in fid patch raptor; do write_directive $m > snartomo_$m.adoc; done
  # batchruntomo, native, full run: the reconstruction and com files the
  # post-reconstruction cases use.
  if [ ! -f brt/TS_01_newstack_full_rec.mrc ]; then
    rm -rf brt; mkdir brt
    cp TS_01_newstack.mrc TS_01_newstack.rawtlt snartomo.adoc brt/
    ( cd brt && nat batchruntomo -RootName TS_01_newstack -CurrentLocation $D/brt \
        -DirectiveFile $D/brt/snartomo.adoc > brt.out 2>&1 ) || echo "native batchruntomo failed; see $D/brt/brt.out" >&2
  fi
  # ruotnocon (snartomo-shared.bash:4404-4530) edits a fiducial model; the
  # fiducialless run above has none, so the e2e pipeline's native patch-
  # tracking model (tiltxcorr, 1024^2, 35 views) stands in.
  [ -f TS_01_newstack.fid ] || cp $E2E/pipe/nat/06_xcorr_pt/ts.fid TS_01_newstack.fid
  if [ ! -f new_wimp.txt ]; then
    nat convertmod TS_01_newstack.fid wimp.txt > /dev/null
    # remove_contours: drop every 7th contour, the way SNARTomo drops the
    # contours whose residual exceeds mean + 3 SD (sort_residuals.py)
    python3 - <<'PY'
import re
t = open('wimp.txt').read().split('\n')
out, i, n = [], 0, 0
# WIMP: "  Object #:  N" blocks; each block's header line "  # of points: K"
blocks, cur = [], []
for line in t:
    if 'Object #:' in line and cur:
        blocks.append(cur); cur = []
    cur.append(line)
blocks.append(cur)
head, objs = blocks[0], blocks[1:]
if not any('Object #:' in l for l in head):
    keep = [b for k, b in enumerate(objs) if k % 7 != 3]
else:
    objs = blocks; head = []; keep = [b for k, b in enumerate(objs) if k % 7 != 3]
open('new_wimp.txt', 'w').write('\n'.join(sum([head] + keep, [])))
PY
  fi
  zs=$(nat imodinfo -a TS_01_newstack.fid | grep scale | grep -v refcurscale | awk '{print $NF}')
  echo "$zs" > zscale.txt
  [ -f tmp.fid ] || nat wmod2imod -z $zs new_wimp.txt tmp.fid > /dev/null
  cd - > /dev/null
}

# A SNARTomo-style batch directive.  SNARTomo copies the user's directive and
# sets setupset.copyarg.pixel to apix/10 nm (update_adoc,
# snartomo-shared.bash:2095-2150); the rest is what an eTomo batch template
# for a single-axis cryo tilt series holds.  THICKNESS is in unbinned pixels
# (1600 = 189 nm -> 200 at bin 8).  Binning 8 = SNARTOMO_BINNING's default
# (snartomo.bashrc.template:66): 4096 -> 512.  The coarse aligned stack
# (prenewst) is binned 4, 1024 x 1024: that is where tracking happens.
#   write_directive          FIDUCIALLESS (runtime.Fiducials.any.fiducialless
#                            = 1): cross-correlation prealignment only
#   write_directive fid      autofidseed seeding + beadtrack (trackingMethod 0,
#                            seedingMethod 1 = auto-seeding, IMOD
#                            com/directives.csv:79-80), then tiltalign
#   write_directive patch    patch tracking (trackingMethod 1): tiltxcorr
#                            xcorr_pt, imodchopconts into 2 pieces
#   write_directive raptor   RAPTOR (trackingMethod 2) through runraptor on the
#                            coarse aligned stack, 30 markers
# The gold size (10 nm = 21 px at bin 4) is what autofidseed/beadtrack/RAPTOR
# look for; TS_01 has gold.  The per-mode lines follow
# /big/henriksson/realbench/brtmissing/pipe/run_brt.sh.
write_directive() {
  local base=1; [ -n "$1" ] && base=0
  awk -v keep=$base 'keep || !/fiducialless/' <<'EOF'
setupset.copyarg.dual = 0
setupset.copyarg.pixel=0.1179
setupset.copyarg.gold = 10
setupset.copyarg.rotation = -90
setupset.copyarg.userawtlt = 1
setupset.scanHeader = 1
runtime.Preprocessing.any.removeXrays = 1
comparam.eraser.ccderaser.PeakCriterion = 10.
comparam.eraser.ccderaser.DiffCriterion = 8.
comparam.xcorr.tiltxcorr.FilterRadius2 = 0.15
comparam.xcorr.tiltxcorr.FilterSigma1 = 0.03
comparam.xcorr.tiltxcorr.FilterSigma2 = 0.05
comparam.prenewst.newstack.BinByFactor = 4
runtime.Fiducials.any.fiducialless = 1
runtime.AlignedStack.any.binByFactor = 8
comparam.tilt.tilt.THICKNESS = 1600
runtime.Reconstruction.any.useSirt = 0
runtime.Postprocess.any.doTrimvol = 1
runtime.Trimvol.any.reorient = 2
EOF
  case $1 in
    fid) cat <<'EOF'
runtime.Fiducials.any.trackingMethod = 0
runtime.Fiducials.any.seedingMethod = 1
comparam.autofidseed.autofidseed.TargetNumberOfBeads = 25
comparam.autofidseed.autofidseed.TwoSurfaces = 1
comparam.autofidseed.autofidseed.AdjustSizes = 1
comparam.track.beadtrack.RoundsOfTracking = 2
EOF
    ;;
    patch) cat <<'EOF'
runtime.Fiducials.any.trackingMethod = 1
comparam.xcorr_pt.tiltxcorr.SizeOfPatchesXandY = 200,200
comparam.xcorr_pt.tiltxcorr.OverlapOfPatchesXandY = 0.33,0.33
comparam.xcorr_pt.tiltxcorr.IterateCorrelations = 1
runtime.PatchTracking.any.contourPieces = 2
EOF
    ;;
    raptor) cat <<'EOF'
runtime.Fiducials.any.trackingMethod = 2
runtime.RAPTOR.any.useAlignedStack = 1
runtime.RAPTOR.any.numberOfMarkers = 30
EOF
    ;;
  esac
}

# ------------------------------------------------------------------- cases
# name | staged inputs (from $B/data; "cp:" = copied fresh before every run
# because the command writes it, "dir:X" = copy everything in data/X) |
# command line (run with cwd = the case directory) | outputs to compare
# ("*" = every file the command created; "stdout" = also compare stdout,
# where SNARTomo parses it).
BRT="-RootName TS_01_newstack -CurrentLocation \$PWD -DirectiveFile \$PWD/snartomo.adoc"
CASES=$(cat <<EOF
# check_frames / framecalc (snartomo-shared.bash:2193, snartomo-framecalc:202,253): header | grep sections, timed by SNARTomo
header_eer|movie.eer|header movie.eer|stdout
header_tif|movie.tif|header movie.tif|stdout
# gain conversion (snartomo-shared.bash:2060)
tif2mrc_gain|gain.tif|tif2mrc gain.tif gain.mrc|gain.mrc
# restack_micrographs (snartomo-shared.bash:3614); also snartomo-oddeven:478 and heatwave.py:2036
newstack_restack|mics TS_01_mcorr.txt|newstack -filei TS_01_mcorr.txt -ou TS_01_newstack.mrc|TS_01_newstack.mrc
# pixel size (snartomo-shared.bash:3530), edits the stack in place
alterheader_pixel|cp:TS_01_newstack.mrc|alterheader -PixelSize 1.179,1.179,1.179 TS_01_newstack.mrc|TS_01_newstack.mrc
# CTF power spectra stack (snartomo-shared.bash:3531)
newstack_ctfs|ctfs TS_01_ctfs.txt|newstack -filei TS_01_ctfs.txt -ou TS_01_ctfstack.mrcs|TS_01_ctfstack.mrcs
# draw_thumbnails (snartomo-shared.bash:3698-3701), thumb_bin=16, box 512, apix 1.179, res_hi 9 -> 136
binvol_thumb|TS_01_newstack.mrc|binvol -z 1 -x 16 -y 16 TS_01_newstack.mrc TS_01_bin.mrcs|TS_01_bin.mrcs
mrc2tif_thumb|TS_01_bin.mrcs|mkdir -p thumbs && mrc2tif -j TS_01_bin.mrcs thumbs/TS_01_newstack|*
clip_resize_ctf|TS_01_ctfstack.mrcs|clip resize -2d -ox 136 -oy 136 TS_01_ctfstack.mrcs TS_01_ctfstack_center.mrcs|TS_01_ctfstack_center.mrcs
mrc2tif_ctf|TS_01_ctfstack_center.mrcs|mkdir -p thumbs && mrc2tif -j TS_01_ctfstack_center.mrcs thumbs/TS_01_ctfstack_center|*
# stack size check (snartomo-shared.bash:3863)
header_stack|TS_01_newstack.mrc|header TS_01_newstack.mrc|stdout
# wrapper_etomo (snartomo-shared.bash:4083), no laudiseron/ruotnocon: one full run
batchruntomo|cp:TS_01_newstack.mrc TS_01_newstack.rawtlt cp:snartomo.adoc|batchruntomo $BRT|TS_01_newstack.mrc TS_01_newstack.prexf TS_01_newstack.prexg TS_01_newstack.xf TS_01_newstack.tlt TS_01_newstack_ali.mrc TS_01_newstack_full_rec.mrc TS_01_newstack_rec.mrc
# the same run with a user directive that tracks fiducials (SNARTomo --batch_directive):
# autofidseed + beadtrack, patch tracking + imodchopconts, RAPTOR.  Every file the run writes.
batchruntomo_fid|cp:TS_01_newstack.mrc TS_01_newstack.rawtlt cp:snartomo_fid.adoc|batchruntomo ${BRT/snartomo.adoc/snartomo_fid.adoc}|*
batchruntomo_patch|cp:TS_01_newstack.mrc TS_01_newstack.rawtlt cp:snartomo_patch.adoc|batchruntomo ${BRT/snartomo.adoc/snartomo_patch.adoc}|*
batchruntomo_raptor|cp:TS_01_newstack.mrc TS_01_newstack.rawtlt cp:snartomo_raptor.adoc|batchruntomo ${BRT/snartomo.adoc/snartomo_raptor.adoc}|*
# ruotnocon_wrapper (snartomo-shared.bash:4460-4509)
convertmod_fid|TS_01_newstack.fid|convertmod TS_01_newstack.fid wimp.txt|wimp.txt
imodinfo_a|TS_01_newstack.fid|imodinfo -a TS_01_newstack.fid|stdout
wmod2imod_z|new_wimp.txt zscale.txt|wmod2imod -z \$(cat zscale.txt) new_wimp.txt tmp.fid|tmp.fid
imodjoin_r1|TS_01_newstack.fid tmp.fid|imodjoin -r 1 TS_01_newstack.fid tmp.fid out.fid|out.fid
# binned pixel size of the tomogram (snartomo-shared.bash:4832): header | grep Pixel
header_rec|brt/TS_01_newstack_full_rec.mrc|header TS_01_newstack_full_rec.mrc|stdout
# central slice (snartomo-shared.bash:4998-5046): the full_rec's short axis is Y -> "-rx -ny 1"
trimvol_slice|brt/TS_01_newstack_full_rec.mrc|trimvol -rx -ny 1 TS_01_newstack_full_rec.mrc TS_01_newstack_full_rec_slice.mrc|TS_01_newstack_full_rec_slice.mrc
mrc2tif_slice|brt/TS_01_newstack_full_rec.mrc|trimvol -rx -ny 1 TS_01_newstack_full_rec.mrc s.mrc > /dev/null && mrc2tif -j s.mrc TS_01_newstack_full_rec_slice.jpg|TS_01_newstack_full_rec_slice.jpg
# snartomo-oddeven:478-497 / snartomo2isonet:446-461: rerun newst/tilt/trimvol on a restacked half set
submfg_oddeven|dir:brt TS_01_newstack.mrc|submfg newst.com tilt.com trimvol.com|TS_01_newstack_ali.mrc TS_01_newstack_full_rec.mrc TS_01_newstack_rec.mrc
# snartomo-animate:250-294 on the tomogram (short axis Y, bin 2): binvol, trimvol -RotateX, mrc2tif -j
binvol_animate|brt/TS_01_newstack_full_rec.mrc|binvol -y 1 -x 2 -z 2 TS_01_newstack_full_rec.mrc rec_bin.mrc|rec_bin.mrc
trimvol_rotx|brt/TS_01_newstack_full_rec.mrc|trimvol -RotateX TS_01_newstack_full_rec.mrc rec_rot.mrc|rec_rot.mrc
mrc2tif_animate|brt/TS_01_newstack_rec.mrc|mrc2tif -j TS_01_newstack_rec.mrc anim|*
EOF
)

stage() { # dir inputs...
  local d=$1 f; shift
  for f in "$@"; do
    case $f in
      cp:*)  cp -a $D/${f#cp:} $d/ ;;
      dir:*) find $D/${f#dir:} -maxdepth 1 -type f -size -1000k ! -name '*.log' ! -name brt.out -exec cp -a {} $d/ \; ;;
      *)     ln -sfn $D/$f $d/$(basename $f) ;;
    esac
  done
}

median() { sort -g | awk '{a[NR]=$1} END{ if(NR%2) print a[(NR+1)/2]; else printf "%.3f\n", (a[NR/2]+a[NR/2+1])/2 }'; }

run_once() { # side threads dir cmd -> "wall rss cpu rc" ; leaves out.txt err.txt
  # CPU is perf's task-clock with inherited counters: it includes every
  # descendant, also those no ancestor wait()s for (GNU time's user+sys only
  # sees waited-for children, and misses most of a batchruntomo run).
  local side=$1 th=$2 d=$3 cmd=$4 omp=""
  [ "$th" != default ] && omp="OMP_NUM_THREADS=$th"
  ( cd $d && perf stat -e task-clock -x, -o .perf /usr/bin/time -f "%e %M" -o .time \
      env -u LD_LIBRARY_PATH $(side_env $side) $omp timeout 3600 bash -c "$cmd" 2>err.txt </dev/null | cat > out.txt
    exit ${PIPESTATUS[0]} )
  local rc=$?
  local cpu=$(awk -F, '$3=="task-clock"{printf "%.2f", $1/1000}' $d/.perf)
  echo "$(tail -1 $d/.time) $cpu $rc"
}

compare() { # natdir rsdir outputs -> verdict
  local nd=$1 rd=$2 outs=$3 files f v
  if [[ " $outs " == *" * "* ]]; then
    files=$( (cd $nd && find . -type f -newer .stage ! -name .time ! -name .perf ! -name .perfr ! -name out.txt ! -name err.txt; \
              cd $rd && find . -type f -newer .stage ! -name .time ! -name .perf ! -name .perfr ! -name out.txt ! -name err.txt) \
| sed 's|^\./||' | sort -u)
  else
    files=$(echo "$outs" | tr ' ' '\n' | grep -v '^stdout$')
  fi
  local extra=""
  [[ " $outs " == *" stdout "* ]] && extra="out.txt"
  python3 $REPO/scripts/snartomo_compare.py $nd $rd $files $extra
}

# ------------------------------------------------------------------ main
[[ $MODE == setup || $MODE == all ]] && setup_data
[[ $MODE == setup ]] && exit 0

STAMP=$(date +%Y%m%d-%H%M%S)
R=$B/results/$STAMP; mkdir -p $R
{
  echo "# bench_snartomo $STAMP host=$(hostname) cores=$(nproc) imod-rs=$RSHASH ($(cd $REPO && git rev-parse --short HEAD)+worktree)"
  echo "# load at start: $(cut -d' ' -f1-3 /proc/loadavg)"
  [ -n "$NATIVE_SUBST" ] && echo "# NATIVE_SUBST (native side is NOT stock): $NATIVE_SUBST"
  [ -n "$RAPTOR_SEED" ] && echo "# RAPTOR_SEED=$RAPTOR_SEED (RAPTOR -seed on both sides)"
} | tee $R/table.txt
printf "%-18s %-7s %-6s %8s %8s %7s %8s %8s %7s %8s %8s %6s %s\n" CASE THREADS PARITY nat_s rs_s WALL natCPU rsCPU CPU natMB rsMB RSS NOTE | tee -a $R/table.txt

while IFS='|' read -r name inputs cmd outs; do
  [ -z "$name" ] && continue; [[ $name == \#* ]] && continue
  [ -n "$ONLY" ] && ! [[ $name =~ $ONLY ]] && continue
  [ -n "$SKIP" ] && [[ $name =~ $SKIP ]] && continue
  for th in $THREADS; do
    nd=$B/run/$name/$th/nat; rd=$B/run/$name/$th/rs
    reps=$REPS; [[ $MODE == parity ]] && reps=${PARITYREPS:-1}
    for side in nat rs; do eval "W_$side=; C_$side=; M_$side=0; RC_$side="; done
    self=""; [[ $reps -ge 2 && $name =~ ${SELFCHECK:-^batchruntomo} ]] && self=1
    rm -rf $B/run/$name/$th/nat.prev $B/run/$name/$th/rs.prev
    for i in $(seq $reps); do
      for side in nat rs; do
        d=$B/run/$name/$th/$side
        if [[ -n $self && $i -eq $reps && -d $d ]]; then mv $d $d.prev; else rm -rf $d; fi
        mkdir -p $d
        stage $d $inputs; sleep 1; touch $d/.stage; sleep 1
        read t m c rc < <(run_once $side $th $d "$cmd")
        echo -e "$name\t$th\t$side\t$i\t$t\t$c\t$m\t$rc" >> $R/runs.tsv
        eval "W_$side+=\"$t \"; C_$side+=\"$c \"; RC_$side=$rc"
        eval "[ $m -gt \$M_$side ] && M_$side=$m"
      done
    done
    nw=$(echo $W_nat | tr ' ' '\n' | median); rw=$(echo $W_rs | tr ' ' '\n' | median)
    nc=$(echo $C_nat | tr ' ' '\n' | median); rc_=$(echo $C_rs | tr ' ' '\n' | median)
    # Short commands: GNU time's 10 ms resolution says nothing, so take
    # `perf stat -r $SHORTREPS` (mean wall and task-clock, ms resolution)
    # in the case directory, after the parity run.
    precise=""
    if [[ $MODE != parity ]] && awk -v a=$nw -v b=$rw 'BEGIN{exit !(a<0.2 && b<0.2)}'; then
      for side in nat rs; do
        d=$B/run/$name/$th/$side; omp=""; [ "$th" != default ] && omp="OMP_NUM_THREADS=$th"
        ( cd $d && env -u LD_LIBRARY_PATH $(side_env $side) $omp perf stat -r ${SHORTREPS:-20} -e task-clock -o .perfr \
            bash -c "$cmd" </dev/null >/dev/null 2>&1 )
        pw=$(awk '/seconds time elapsed/{printf "%.4f", $1}' $d/.perfr)
        pc=$(awk '/task-clock/{gsub(",","",$1); printf "%.4f", $1/1000}' $d/.perfr)
        eval "PW_$side=$pw; PC_$side=$pc"
      done
      nw=$PW_nat; rw=$PW_rs; nc=$PC_nat; rc_=$PC_rs; precise=" [perf -r ${SHORTREPS:-20}]"
    fi
    par=$(compare $nd $rd "$outs"); verdict=${par%%$'\t'*}; note=${par#*$'\t'}
    [ "$RC_nat" != "$RC_rs" ] && { verdict=RC; note="rc $RC_nat vs $RC_rs; $note"; }
    echo -e "$name\t$th\t$verdict\t$note" >> $R/parity.tsv
    lim=0.05; [ -n "$precise" ] && lim=0
    sp=$(awk  -v n=$nw -v r=$rw -v l=$lim 'BEGIN{print (n<l||r<l||r==0) ? "-" : sprintf("%.2fx", n/r)}')
    cpr=$(awk -v n=$nc -v r=$rc_ -v l=$lim 'BEGIN{print (n<l||r<l||r==0) ? "-" : sprintf("%.2fx", n/r)}')
    rr=$(awk -v n=$M_nat -v r=$M_rs 'BEGIN{printf "%.2fx", (n>0? r/n : 0)}')
    printf "%-18s %-7s %-6s %8s %8s %7s %8s %8s %7s %8.1f %8.1f %6s %s\n" "$name" "$th" "$verdict" \
      "$nw" "$rw" "$sp" "$nc" "$rc_" "$cpr" $(awk -v v=$M_nat 'BEGIN{print v/1024}') \
      $(awk -v v=$M_rs 'BEGIN{print v/1024}') "$rr" "$( [ $verdict = IDENT ] || echo "${note:0:120}")$precise" | tee -a $R/table.txt
    if [ -n "$self" ]; then
      # each side against itself (previous rep vs last): run-to-run variation
      for side in nat rs; do
        d=$B/run/$name/$th/$side; ps_=$(compare $d.prev $d "$outs")
        echo -e "$name\t$th\tSELF-$side:${ps_%%$'\t'*}\t${ps_#*$'\t'}" >> $R/parity.tsv
        echo "#   $name $th $side-vs-$side: ${ps_%%$'\t'*} ${ps_#*$'\t'}" | cut -c1-200 | tee -a $R/table.txt
      done
    fi
  done
done <<< "$CASES"
echo "# load at end: $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a $R/table.txt
echo "results: $R" >&2
