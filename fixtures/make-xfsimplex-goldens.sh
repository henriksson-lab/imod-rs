#!/bin/bash
# Regenerate fixtures/xfsimplex/golden/ from the native reference xfsimplex (see
# make-small-prog-goldens.sh for the layout).
#
# Exception (BUGS.md, xfsimplex, fixed in translation): native passes the
# integer 2 as scaledSobel's real centre weight, a denormal; the translation
# uses 2.0.  golden/s_sobel.{stdout,out.o.xf} are therefore written by the
# translation (target/release/imod xfsimplex, build it first), after the
# native run.
set -e
"$(dirname "$0")/make-small-prog-goldens.sh" xfsimplex flib/image/xfsimplex
F=$(cd "$(dirname "$0")/xfsimplex" && pwd)
S=$(mktemp -d)
cp "$F"/*.mrc "$F"/*.xf "$S"/
(cd "$S" && AUTODOC_DIR=$F/../../IMOD/autodoc "$F/../../target/release/imod" xfsimplex \
    -aimage a.mrc -bimage b.mrc -output o.xf -sobel -binning 1 -variables 3 < /dev/null \
    | cat > "$F/golden/s_sobel.stdout")
cp "$S/o.xf" "$F/golden/s_sobel.out.o.xf"
rm -rf "$S"
