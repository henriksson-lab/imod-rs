#!/bin/bash
# Regenerate fixtures/pickbestseed/golden from the native reference
# pickbestseed.
#
# Inputs in fixtures/pickbestseed come from a native autofidseed run
# (`-track track.com -number 40 -two -leave`) on the TS_01 prealigned stack
# of the e2e-ts01 pipeline (1024x1024x35, beads of 21.2 pixels), 2026-09-27:
# t0-t2.mod are the three tracked seed models (seeds from views 17-19),
# e0-e2.txt the elongation files and s0-s2.txt the surface files beadtrack and
# sortbeadsurfs wrote for them, seedin.mod the seed model that run produced,
# and base.in the pickbestseed input autofidseed composed.  bound.mod is a
# two-contour boundary model on Z 17 written by the native point2model.  Each
# row of cases.tsv drops lines of base.in by prefix and appends others; the
# case runs in its own directory (with seedin.mod as seed.mod, which
# -AppendToSeedModel reads) with that input on stdin and stdout captured
# through a pipe.  golden/<case>.rc is the exit status, .out the standard
# output, golden/<case>/ every file the run wrote.
#
# `make-pickbestseed-goldens.sh defined` instead writes defined/ from the
# fixed translation ($RSBIN, default target/release/imod) for the cases named
# in defined.list -- those whose output an upstream-bug fix changes (BUGS.md,
# pickbestseed); golden/ is untouched.
set -e
REF=${REF:-/tmp/imod-reference-build}
HERE=$(cd "$(dirname "$0")/pickbestseed" && pwd)
export AUTODOC_DIR=$REF/autodoc LD_LIBRARY_PATH=$REF/buildlib
PROG=$REF/imodutil/pickbestseed
DEST=golden
if [ "$1" = defined ]; then
  RSBIN=${RSBIN:-$(cd "$HERE/../.." && pwd)/target/release/imod}
  PROG="$RSBIN pickbestseed"
  DEST=defined
  export AUTODOC_DIR=$(cd "$HERE/../.." && pwd)/IMOD/autodoc
fi
rm -rf "$HERE/$DEST"; mkdir -p "$HERE/$DEST"
sed "${FULL:+s/^#full\t//;}/^#/d" "$HERE/cases.tsv" | while IFS=$'\t' read -r name drop add; do
  [ -z "$name" ] && continue
  [ $DEST = defined ] && ! grep -qx "$name" <(grep -v '^#' "$HERE/defined.list" | cut -f1) && continue
  work=$(mktemp -d)
  cp "$HERE"/t?.mod "$HERE"/e?.txt "$HERE"/s?.txt "$HERE"/bound.mod "$work"/
  cp "$HERE/seedin.mod" "$work/seed.mod"
  { cat "$HERE/e0.txt"; echo "  1    999     0.5000    0.5000    0.5000     0.2000     0.3000     0.3000     0.3000       0.00"; } > "$work/badco.txt"
  ls -A "$work" > "$work/.inputs"
  # The case's standard input: base.in without the lines starting with any
  # comma-separated prefix in $drop, then the ';'-separated lines of $add.
  awk -v drop="$drop" 'BEGIN { n = (drop == "-") ? 0 : split(drop, d, ",") }
    { for (i = 1; i <= n; i++) if (index($0, d[i]) == 1) next; print }' "$HERE/base.in" > "$work/.stdin"
  [ "$add" = "-" ] || echo "$add" | tr ';' '\n' >> "$work/.stdin"
  set +e
  (cd "$work" && timeout 20 $PROG -StandardInput < .stdin | cat > "$work/.stdout"; echo "${PIPESTATUS[0]}" > "$work/.rc")
  set -e
  mkdir -p "$HERE/$DEST/$name"
  cp "$work/.stdout" "$HERE/$DEST/$name.out"
  cp "$work/.rc" "$HERE/$DEST/$name.rc"
  for f in "$work"/*; do
    b=$(basename "$f")
    grep -qx "$b" "$work/.inputs" || cp "$f" "$HERE/$DEST/$name/"
  done
  # seed.mod is an input that the run replaces (renaming it to seed.mod~)
  if [ -e "$work/seed.mod~" ]; then cp "$work/seed.mod" "$HERE/$DEST/$name/"; fi
  rm -rf "$work"
done
