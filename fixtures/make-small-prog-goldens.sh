#!/bin/bash
# Shared body of make-{extracttilts,extractmagrad,xcorrstack,tomopitch}-goldens.sh.
# Usage: make-small-prog-goldens.sh <program> <path of native binary under the
# reference build>.  Every row of fixtures/<program>/cases.tsv (name, arguments
# or -, stdin as printf escapes or -) runs natively in a fresh directory holding
# copies of the fixture inputs, with stdout captured through a pipe.
# golden/<name>.rc is the exit status, .stdout the standard output, and
# golden/<name>.out.<file> each file the run created or changed (an output, or
# the `~` backup of an existing file it replaced).
# An optional fourth column gives native-equivalent arguments: a case that
# reaches an upstream bug fixed in the translation (BUGS.md) is run natively
# with these arguments instead -- an input that does not trigger the bug and
# whose output is the behaviour the fix defines -- so its golden is still a
# native output.  An optional fifth column is a sed expression applied to that
# run's standard output, for a name the equivalent input had to change (for
# example a copy of a file, which the program echoes when it opens it).
set -e
P=$1
R=${IMOD_REF:-/tmp/imod-reference-build}
NATIVE=$R/$2
F=$(cd "$(dirname "$0")/$P" && pwd)
S=$(mktemp -d)
mkdir -p "$F/golden"
rm -f "$F"/golden/*
# SHARED: extra inputs another suite owns, as paths relative to fixtures/<program>.
inputs() {
  find "$F" -maxdepth 1 -type f ! -name cases.tsv ! -name 'make-*' ! -name golden.manifest
  for s in $SHARED; do echo "$F/$s"; done
}
sed "${FULL:+s/^#full\t//;}/^#/d" "$F/cases.tsv" | while IFS=$'\t' read -r name args stdin nat sub; do
  [ -z "$name" ] && continue
  [ -n "$nat" ] && [ "$nat" != "-" ] && args=$nat
  [ "$args" = "-" ] && args=""
  [ "$stdin" = "-" ] && stdin=""
  d="$S/$name"; mkdir "$d"
  inputs | xargs -I{} cp {} "$d"/
  set +e
  (cd "$d" && printf -- "$stdin" | AUTODOC_DIR=$R/autodoc LD_LIBRARY_PATH=$R/buildlib \
      timeout 60 $NATIVE $args 2>/dev/null | cat > "$S/stdout"; exit ${PIPESTATUS[1]})
  echo $? > "$F/golden/$name.rc"; set -e
  if [ -n "$sub" ] && [ "$sub" != "-" ]; then sed -e "$sub" "$S/stdout" > "$F/golden/$name.stdout"
  else cp "$S/stdout" "$F/golden/$name.stdout"; fi
  for f in "$d"/*; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    src=$F/$b
    for s in $SHARED; do [ "$(basename "$s")" = "$b" ] && src=$F/$s; done
    if [ ! -f "$src" ] || ! cmp -s "$f" "$src"; then cp "$f" "$F/golden/$name.out.$b"; fi
  done
done
rm -rf "$S"
