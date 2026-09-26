#!/bin/bash
# Regenerate fixtures/fakevolume/golden/ from the native reference.  For each
# row of cases.tsv the native program runs in a fresh directory with stdout
# captured through a pipe; golden/<name>.rc is the exit status, .stdout the
# standard output and .o.mrc the output volume when native left one.
# Cases that read uninitialised stack in the source (-trunc entered fewer
# times than there are cylinders with no spheres, fakevolume.cpp:199; see
# BUGS.md) are deliberately absent: native is not stable across runs there.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
F=$(cd "$(dirname "$0")/fakevolume" && pwd)
export AUTODOC_DIR=$R/autodoc LD_LIBRARY_PATH=$R/buildlib
rm -rf "$F/golden"; mkdir -p "$F/golden"
sed "${FULL:+s/^#full\t//;}/^#/d" "$F/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ -z "$name" ] && continue
  S=$(mktemp -d); cd "$S"
  set +e
  $R/mrc/fakevolume $args < /dev/null | cat > "$F/golden/$name.stdout"
  echo "${PIPESTATUS[0]}" > "$F/golden/$name.rc"
  set -e
  [ -f o.mrc ] && cp o.mrc "$F/golden/$name.o.mrc"
  cd /; rm -rf "$S"
done
# Upstream message typos fixed in translation (BUGS.md, fakevolume): the goldens
# carry the corrected option names.
sed -i 's/You cannot enter -cdens, -cradii, or -ctype without/You cannot enter -cdens, -cradii, or -trunc without/' "$F/golden/e_nocyl.stdout"
sed -i 's/You must enter -ctype, -cdens, and -cradii for all spheres/You must enter -stype, -sdens, and -sradii for all spheres/' "$F/golden/e_mixtype.stdout"
