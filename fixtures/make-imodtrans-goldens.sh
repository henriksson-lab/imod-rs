#!/bin/bash
# Regenerate fixtures/imodtrans/golden/ from the native reference imodtrans.
# Inputs: multi.mod = native wmod2imod of multi.wimp; meshed.mod = native
# imodmesh of it; flipped.mod = native `imodtrans -T multi.mod`; img1/img2.mrc
# = native raw2mrc 8x6x2 float, then alterheader org/del(/tlt) as in
# tests/imodtrans_cli.rs.  Each case writes golden/<name>.mod (when native
# leaves an output file) from `imodtrans <args> in out.mod`.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
F=$(cd "$(dirname "$0")/imodtrans" && pwd)
S=$(mktemp -d)
cp "$F"/*.mod "$F"/*.xf "$F"/*.mrc "$S"/
cp "$F"/../../IMOD/Etomo/uitestData/BB/{BBa_erase.fid,BBa.xf} "$S"/
rm -f "$F"/golden/*
# An optional fifth column gives the arguments native is run with instead:
# `-2 f -l N` adds -tx/-ty twice natively (BUGS.md, fixed in translation), so
# the golden for the translation's `-tx 1.5 -ty 2` is native's `-tx 0.75 -ty 1`.
sed "${FULL:+s/^#full\t//;}/^#/d" "$F/cases.tsv" | while IFS=$'\t' read -r name input args rc nargs; do
  [ -n "$nargs" ] && args=$nargs
  d="$S/$name"; mkdir "$d"; cp "$S"/*.xf "$S"/*.mrc "$d"/; cp "$S/$input" "$d/in"
  set +e
  (cd "$d" && AUTODOC_DIR=$R/autodoc LD_LIBRARY_PATH=$R/buildlib $R/imodutil/imodtrans $args in out.mod | cat >/dev/null; exit ${PIPESTATUS[0]})
  got=$?; set -e
  [ "$got" = "$rc" ] || { echo "$name: native exit $got, table says $rc"; exit 1; }
  if [ -f "$d/out.mod" ]; then cp "$d/out.mod" "$F/golden/$name.mod"; fi
done
rm -rf "$S"
