#!/bin/bash
# Regenerate fixtures/autofidseed/golden from the native Python autofidseed
# (IMOD/pysrc/autofidseed) driving OUR programs.  The inputs are a 128 x 128 x
# 11 byte crop (binned by 2) of the prealigned TS_01 stack around zero tilt,
# with gold beads, its tilt angles, a beadtrack command file made from the
# copytomocoms track.com (ImageFile/TiltFile/BeadDiameter/PixelSize/box size
# adjusted to the crop, no prealignment transforms), and a boundary model.
#
# Why our programs: every program autofidseed runs (imodfindbeads, imodmop,
# newstack, tiltxcorr, clipmodel, beadtrack, point2model, sortbeadsurfs,
# pickbestseed, header) is translated and has its own native-golden suite;
# several carry fixed upstream bugs (BUGS.md) and native beadtrack is not
# deterministic from run to run, so the native programs cannot give a stable
# golden for the script.  With our programs on both sides, these goldens
# test exactly the script's translation: its option handling, the commands
# and input it composes, the files it writes, keeps and removes, and its
# messages.  REF defaults to /tmp/imod-reference-build; the imod binary to
# target/release/imod.
#
# Per case: golden/<case>.rc, .out (standard output through a pipe),
# .files (every file left, with the PID in temporary names replaced by PID),
# and golden/<case>/<file> for every file that is not an unchanged input.
set -e
REF=${REF:-/tmp/imod-reference-build}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
F=$ROOT/fixtures/autofidseed
IMOD=${IMOD_BIN:-$ROOT/target/release/imod}
S=$(mktemp -d)
mkdir $S/bin
for c in imodfindbeads imodmop newstack tiltxcorr clipmodel beadtrack point2model \
         sortbeadsurfs pickbestseed header; do
  ln -s $IMOD $S/bin/$c
done
export AUTODOC_DIR=$ROOT/IMOD/autodoc IMOD_DIR=$REF PYTHONPATH=$ROOT/IMOD/pysrc \
  PATH=$S/bin:$PATH
unset IMOD_OUTPUT_FORMAT PARALLEL_BOUNDARY_SIZE RUNCMD_VERBOSE
rm -rf $F/golden; mkdir $F/golden
sed "${FULL:+s/^#full\t//;}/^#/d" $F/cases.tsv | while IFS=$'\t' read -r name first args; do
  d=$S/run/$name; mkdir -p $d; cd $d
  cp $F/track.com $F/small.st $F/small.rawtlt $F/bound.mod .
  if [ "$first" != "-" ]; then
    eval "set -- $first"; python3 $ROOT/IMOD/pysrc/autofidseed "$@" > /dev/null 2>&1 || true
  fi
  eval "set -- $args"
  set +e; python3 $ROOT/IMOD/pysrc/autofidseed "$@" 2>/dev/null | cat > $S/out; rc=${PIPESTATUS[0]}; set -e
  echo $rc > $F/golden/$name.rc; cp $S/out $F/golden/$name.out
  for f in $(find . -type f | sed 's|^\./||' | sort); do
    m=$(echo $f | sed 's/afs[0-9]*\./afsPID./')
    echo $m >> $F/golden/$name.files
    if [ -e $F/$f ] && cmp -s $f $F/$f; then continue; fi
    mkdir -p $(dirname $F/golden/$name/$m); cp $f $F/golden/$name/$m
  done
done
rm -rf $S
