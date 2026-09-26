#!/bin/bash
# Regenerate fixtures/modifymdoc/golden from the native reference modifymdoc.
# The inputs are hand-authored SerialEM-style .mdoc files (tilt series with
# tied tilts and DateTime entries, montage, frame set, image series, NaN and
# missing tilts, bad entries); p.param is a PIP parameter file.  Each case in
# modifymdoc/cases.tsv runs in its own directory (which also holds an empty
# directory "sub") with stdout captured through a pipe and TZ=UTC, since the
# dose ordering goes through mktime.  "." in the arguments column means none.
# Exit status goes to golden/<case>.rc, stdout to golden/<case>.out, and every
# new or changed file to golden/<case>/.
#
# PROG=<binary> OUT=<dir> runs another binary (e.g. "imod modifymdoc" through
# a link) into another golden directory, for a direct differential.
set -e
REF=${REF:-/tmp/imod-reference-build}
PROG=${PROG:-$REF/mrc/modifymdoc}
export AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib TZ=UTC
HERE=$(cd "$(dirname "$0")/modifymdoc" && pwd)
OUT=${OUT:-$HERE/golden}
rm -rf "$OUT"; mkdir -p "$OUT"
sed "${FULL:+s/^#full\t//;}/^#/d" "$HERE/cases.tsv" | while IFS=$'\t' read -r name args; do
  [ -z "$name" ] && continue
  [ "$args" = "." ] && args=
  work=$(mktemp -d)
  cp "$HERE"/*.mdoc "$HERE"/p.param "$work"/
  mkdir "$work/sub"
  set +e
  (cd "$work" && timeout 20 $PROG $args < /dev/null | cat > "$work/.stdout"; echo "${PIPESTATUS[0]}" > "$work/.rc")
  set -e
  mkdir -p "$OUT/$name"
  cp "$work/.stdout" "$OUT/$name.out"
  cp "$work/.rc" "$OUT/$name.rc"
  for f in "$work"/*; do
    b=$(basename "$f")
    [ -d "$f" ] && continue
    if [ -f "$HERE/$b" ] && cmp -s "$f" "$HERE/$b"; then continue; fi
    cp "$f" "$OUT/$name/"
  done
  rm -rf "$work"
done
# Upstream bugs fixed in translation (BUGS.md, modifymdoc).  Native exits 1
# on `-dose` for an mdoc without DateTime entries (it treats the documented
# -1 "no time stamps, file order assumed" return as failure); the translation
# writes the output.  golden/{dosenotime,scrambleddose}.{rc,out} and their
# out.mdoc were written by the translation and verified equal to native run
# on the same mdoc with DateTime entries added in file order, then removed.
# The missing-file message's "dose not exist" typo is corrected.
if [ "$PROG" = "$REF/mrc/modifymdoc" ]; then
  sed -i 's/dose not exist/does not exist/' "$OUT/nonexist.out"
  echo "NOTE: restore golden/dosenotime* and golden/scrambleddose* from git (translation-written)"
fi
