#!/bin/bash
# Regenerate fixtures/xfmodel/golden/ from the native reference xfmodel.
# Inputs: the files in fixtures/xfmodel (seeded linear and inverse-grid warping
# transforms in warpfiles.c's version-2/3 layouts, distortion fields, a
# mag-gradient file, piece lists, small byte MRC images written by native
# raw2mrc, and m1.mod/m2.mod written by native wmod2imod from w1.wimp), plus
# BBa_erase.fid, BBa.xf and BBb.xf from the vendored IMOD/Etomo/uitestData/BB.
# For each row of cases.tsv the native program runs in a fresh directory with
# stdout captured through a pipe; golden/<name>.rc holds the exit status,
# .stdout the standard output and .out the output file (o.mod or o.xf) when
# native leaves one.  OMP_NUM_THREADS=1 only saves time: extrapolateGrid's
# OpenMP split does not change its result.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
F=$(cd "$(dirname "$0")/xfmodel" && pwd)
V="$F/../../IMOD/Etomo/uitestData/BB"
S=$(mktemp -d)
rm -f "$F"/golden/*
grep -v '^#' "$F/cases.tsv" | while IFS=$'\t' read -r name args stdin; do
  [ "$args" = "-" ] && args=""
  d="$S/$name"; mkdir "$d"
  find "$F" -maxdepth 1 -type f ! -name cases.tsv -exec cp {} "$d"/ \;
  cp "$V/BBa_erase.fid" "$V/BBa.xf" "$V/BBb.xf" "$d"/
  [ "$stdin" = "-" ] && stdin=""
  set +e
  (cd "$d" && printf -- "$stdin" | OMP_NUM_THREADS=1 AUTODOC_DIR=$R/autodoc \
      LD_LIBRARY_PATH=$R/buildlib timeout 120 $R/flib/model/xfmodel $args 2>/dev/null \
      | cat > "$S/stdout"; exit ${PIPESTATUS[1]})
  echo $? > "$F/golden/$name.rc"; set -e
  cp "$S/stdout" "$F/golden/$name.stdout"
  for o in o.mod o.xf; do
    if [ -f "$d/$o" ]; then cp "$d/$o" "$F/golden/$name.out"; fi
  done
done
rm -rf "$S"
