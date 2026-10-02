#!/bin/bash
# Regenerate fixtures/raptor/golden from the native reference RAPTOR.
#
# Input: raptor/syn1.mrc is a seeded synthetic tilt series (128 x 128, 21 views at
# -60..60 in 6-degree steps, 18 dark Gaussian beads, sigma 2.2, projected from random
# 3D positions with a 4-degree tilt axis, per-view shifts and noise), written as floats
# by the native raw2mrc and converted to bytes by the native `newstack -mode 0 -scale
# 0,255` (`make-syn1.py syn1 128 128 -60 60 6 18 11 2.2`); syn1.rawtlt its angles.
#
# Each row of cases.tsv (name, RAPTOR options, a fixture file to delete first or `-`)
# runs `RAPTOR -exec <dir of MarkersCorrespond> -path . -inp syn1.mrc -out r <options>`
# in a fresh directory with the fixture inputs, stdout captured through a pipe; the exit
# status goes to golden/<case>.rc, stdout to golden/<case>.out and every file under r/
# to golden/<case>/.  The non-track case runs the native tiltalign and newstack from
# $NATBIN (a directory of native IMOD programs, default the e2e harness's).
#
# `make-raptor-goldens.sh defined` then overwrites golden/<case> with the fixed
# translation's output ($RSBIN, default target/release/imod) for the cases named in
# defined.list -- those an upstream-bug fix changes (BUGS.md, RAPTOR, 2026-09-27).
# The main one: the `.cfg` marker count RAPTOR hands MarkersCorrespond is one short
# whenever a previous projection pair supplies M+1 reference markers, and
# MarkersCorrespond then reads its per-marker arrays past their end; that changes
# the beliefs, and so the model, of essentially every run.  Set PATCHED to a native
# RAPTOR built from IMOD/raptor with the defined behaviour patched in (the cfg count,
# LPsolver's copy, STDQPdata's initial values, and the backward-trajectory seed:
# raptor/native-defined.patch, applied
# with `patch -p1` in a copy of IMOD/raptor, built with the reference CXXFLAGS and
# linked to the reference libopencv.a/libcsparse.a) to check the defined goldens
# against it instead of trusting our build: `make-raptor-goldens.sh patched` writes
# raptor/patched/, to be compared with golden/ (log dates, argv line and timing
# seconds masked).  On 2026-09-27 every file matched except the exit status of
# notrack/nortlt (exit-status fix, not in the patch), notrack's tiltalign log
# (a trailing blank line from our in-process tiltalign) and edgepeak (native reads
# past an image there).
set -e
REF=${REF:-/tmp/imod-reference-build}
NATBIN=${NATBIN:-/big/henriksson/realbench/e2e-ts01/natimod/bin}
HERE=$(cd "$(dirname "$0")/raptor" && pwd)
export AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib PATH=$NATBIN:$PATH
PROG="$REF/raptor/RAPTOR"
EXEC=$REF/raptor
OUT=golden
case "$1" in
  defined)
    RSBIN=${RSBIN:-$(cd "$HERE/../.." && pwd)/target/release/imod}
    PROG="$RSBIN RAPTOR"
    export AUTODOC_DIR=$(cd "$HERE/../.." && pwd)/IMOD/autodoc ;;
  patched)
    PROG=${PATCHED:?set PATCHED to a patched native RAPTOR}
    OUT=patched ;;
  *) rm -rf "$HERE/golden"; mkdir -p "$HERE/golden" ;;
esac
grep -v '^#' "$HERE/cases.tsv" | while IFS=$'\t' read -r name opts drop; do
  if [ "$1" = defined ] || [ "$1" = patched ]; then
    grep -qx "$name" <(grep -v '^#' "$HERE/defined.list" | cut -f1) || continue
    rm -rf "$HERE/$OUT/$name" "$HERE/$OUT/$name.out" "$HERE/$OUT/$name.rc"
  fi
  mkdir -p "$HERE/$OUT"
  work=$(mktemp -d)
  cp "$HERE"/syn1.mrc "$HERE"/syn1.rawtlt "$work"/
  [ "$drop" = - ] || rm -f "$work/$drop"
  set +e
  (cd "$work" && $PROG -exec $EXEC -path . -inp syn1.mrc -out r $opts | cat > "$work/.stdout"; echo "${PIPESTATUS[0]}" > "$work/.rc")
  set -e
  mkdir -p "$HERE/$OUT/$name"
  cp "$work/.stdout" "$HERE/$OUT/$name.out"
  cp "$work/.rc" "$HERE/$OUT/$name.rc"
  [ -d "$work/r" ] && (cd "$work/r" && cp --parents -r . "$HERE/$OUT/$name/")
  rm -rf "$work"
done
