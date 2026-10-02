#!/bin/bash
# Regenerate fixtures/runraptor/golden/ from the native Python runraptor
# (IMOD/pysrc/runraptor, run from the reference install) with the native
# header and xfmodel on PATH.
#
# RAPTOR itself is the stand-in fixtures/runraptor/RAPTOR (RAPTOR_BIN points
# at fixtures/runraptor), which writes what runraptor reads from a RAPTOR run:
# the fiducial text model stub.fid.txt (real RAPTOR output for the TS_01
# stack) and the log.  The real-RAPTOR differential is in TODO.md
# ("runraptor").  Inputs:
#  stub.mrc     header-only 512 x 512 x 35 float MRC, pixel 9.432/9.432/2.358
#  stub.prexg   the TS_01 e2e pipeline's coarse transforms
#  pn_*.com     prenewst.com variants: -StandardInput with ImagesAreBinned,
#               a newstack command line, a blank parameter line (defined
#               case), and a missing DistortionField plus ImagesAreBinned and
#               GradientFile, as parameter lines and as newstack options
#               (xfmodel fails, so its command line shows what was parsed)
# Each cases.tsv row is name<TAB>setup<TAB>arguments (setup tokens: src:dst
# copies a fixture, dir:d makes a directory, old:f writes a file, rbin:none
# sets RAPTOR_BIN to a missing directory).  golden/<case>.rc is the exit
# status, .out the standard output (through a pipe), and golden/<case>/ every
# file that is new or changed afterwards, its path with / written as %.  The
# temporary model runraptor names <root>.<pid> is renamed <root>.PID; in the
# output the PID is masked the same way and the RAPTOR_BIN directory reads
# RAPTOR_BIN.
#
# `make-runraptor-goldens.sh defined` writes defined/ instead, from our own
# build ($RSBIN, default target/release/imod) for the cases in defined.list,
# whose output an upstream-bug fix changes (BUGS.md).
set -e
REF=${REF:-/tmp/imod-reference-build}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HERE=$ROOT/fixtures/runraptor
DEST=golden
BIN=$(mktemp -d)
mkdir -p $BIN/bin
if [ "$1" = defined ]; then
  DEST=defined
  RSBIN=${RSBIN:-$ROOT/target/release/imod}
  for c in runraptor xfmodel header; do ln -s $RSBIN $BIN/bin/$c; done
  export AUTODOC_DIR=$ROOT/IMOD/autodoc
else
  ln -s $REF/pysrc $BIN/pylib
  ln -s $REF/pysrc/runraptor $BIN/bin/runraptor
  ln -s $REF/flib/model/xfmodel $BIN/bin/xfmodel
  ln -s $REF/flib/image/header $BIN/bin/header
  export AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib
fi
export IMOD_DIR=$BIN PATH=$BIN/bin:$PATH
rm -rf "$HERE/$DEST"; mkdir -p "$HERE/$DEST"
sed "${FULL:+s/^#full\t//;}/^#/d" "$HERE/cases.tsv" | while IFS=$'\t' read -r name setup args; do
  [ -z "$name" ] && continue
  [ $DEST = defined ] && ! grep -qx "$name" <(grep -v '^#' "$HERE/defined.list" | cut -f1) && continue
  work=$(mktemp -d)
  rbin=$HERE
  [ "$setup" = - ] && setup=""
  for tok in $setup; do
    case $tok in
      dir:*) mkdir -p "$work/${tok#dir:}" ;;
      old:*) echo old > "$work/${tok#old:}" ;;
      rbin:none) rbin=$work/nothere ;;
      *) cp "$HERE/${tok%%:*}" "$work/${tok#*:}" ;;
    esac
  done
  (cd "$work" && find . -type f | sort | while read -r f; do sha256sum "$f"; done) > "$work/../.inputs.$$"
  set +e
  (cd "$work" && RAPTOR_BIN=$rbin timeout 300 runraptor $args 2>/dev/null | cat > "$work/../.stdout.$$";
   echo "${PIPESTATUS[0]}" > "$work/../.rc.$$")
  set -e
  mkdir -p "$HERE/$DEST/$name"
  sed "s,$rbin,RAPTOR_BIN,g; s/ts\\.[0-9][0-9]*/ts.PID/g" "$work/../.stdout.$$" > "$HERE/$DEST/$name.out"
  cp "$work/../.rc.$$" "$HERE/$DEST/$name.rc"
  (cd "$work" && find . -type f | sort) | while read -r f; do
    grep -qx "$(cd $work && sha256sum "$f")" "$work/../.inputs.$$" && continue
    flat=$(echo "${f#./}" | sed 's/^ts\.[0-9][0-9]*$/ts.PID/; s,/,%,g')
    cp "$work/$f" "$HERE/$DEST/$name/$flat"
  done
  rm -rf "$work" "$work/../.inputs.$$" "$work/../.stdout.$$" "$work/../.rc.$$"
done
rm -rf $BIN
