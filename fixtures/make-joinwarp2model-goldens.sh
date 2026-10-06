#!/bin/bash
# Regenerate fixtures/joinwarp2model/golden/ from the native reference joinwarp2model
# (see make-small-prog-goldens.sh for the layout).  Inputs: make-etomo-prog-inputs.sh.
#
# The program runs `xfmodel` through system().  The goldens are made with OUR
# xfmodel first on PATH (a link to the crate's `imod` binary): xfmodel's grid
# extension is an upstream bug fixed in translation (BUGS.md, xfmodel, "Model
# range extended by + instead of *"), which moves warp-extrapolated points, so
# native joinwarp2model over our xfmodel is the native-equivalent of the
# defined behaviour.  Build first (cargo build --release).
R=${IMOD_REF:-/tmp/imod-reference-build}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OURS=${IMOD_RS_BIN:-${CARGO_TARGET_DIR:-$ROOT/target}/release/imod}
[ -x "$OURS" ] || { echo "no imod binary at $OURS"; exit 1; }
B=$(mktemp -d)
ln -s "$OURS" "$B/xfmodel"
export PATH=$B:$PATH
"$(dirname "$0")/make-small-prog-goldens.sh" joinwarp2model imodutil/joinwarp2model
rc=$?
rm -rf "$B"
exit $rc
