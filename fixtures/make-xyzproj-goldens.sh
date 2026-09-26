#!/bin/bash
# Regenerate fixtures/xyzproj/golden/ from the native reference xyzproj (see
# make-small-prog-goldens.sh for the layout), then regenerate the cases listed
# in xyzproj/make-fixed-cases.txt from imod-rs: those reach upstream defects
# the translation fixes (BUGS.md), so native is not their reference.
set -e
D=$(cd "$(dirname "$0")" && pwd)
# ONLY_FIXED=1 skips the native run and regenerates just the listed cases.
[ -n "$ONLY_FIXED" ] || "$D/make-small-prog-goldens.sh" xyzproj flib/image/xyzproj
R=${IMOD_REF:-/tmp/imod-reference-build}
RSBIN=${RSBIN:-$D/../target/release/imod}
F=$D/xyzproj
S=$(mktemp -d)
for name in $(grep -v '^#' "$F/make-fixed-cases.txt"); do
  line=$(grep -P "^$name\t" "$F/cases.tsv")
  IFS=$'\t' read -r _ args stdin <<< "$line"
  [ "$args" = "-" ] && args=""
  [ "$stdin" = "-" ] && stdin=""
  rm -f "$F"/golden/$name.*
  d="$S/$name"; mkdir "$d"
  find "$F" -maxdepth 1 -type f ! -name cases.tsv ! -name 'make-*' ! -name golden.manifest -exec cp {} "$d"/ \;
  set +e
  (cd "$d" && printf -- "$stdin" | AUTODOC_DIR=$R/autodoc \
      timeout 60 "$RSBIN" xyzproj $args 2>/dev/null | cat > "$S/stdout"; exit ${PIPESTATUS[1]})
  echo $? > "$F/golden/$name.rc"; set -e
  cp "$S/stdout" "$F/golden/$name.stdout"
  for f in "$d"/*; do
    b=$(basename "$f")
    if [ ! -f "$F/$b" ] || ! cmp -s "$f" "$F/$b"; then cp "$f" "$F/golden/$name.out.$b"; fi
  done
done
rm -rf "$S"
