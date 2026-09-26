#!/bin/bash
# Regenerate fixtures/jpeg/golden/ from the native reference.  The JPEG inputs
# were authored with Pillow (seeded gradients + noise; gray, 4:2:0, 4:4:4,
# 4:2:2, progressive), ImageMagick (cam.jpg: 2x1 sampling with a comment;
# cmyk.jpg), cjpeg (rst.jpg: -restart 1) and native `mrc2tif -j`
# (mtb.jpg, mtrgb.jpg).
#
# The reference build is configured NO_HDF_LIB, whose iiHDFCheck stub returns
# IIERR_NO_SUPPORT and so ends iiOpen's check loop before iiJPEGCheck (BUGS.md).
# jpegfirst.so, built here, is an LD_PRELOAD shim that inserts iiJPEGCheck
# ahead of it -- the order an HDF-enabled build effectively has.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
F=$(cd "$(dirname "$0")/jpeg" && pwd)
export AUTODOC_DIR=$R/autodoc LD_LIBRARY_PATH=$R/buildlib
W=$(mktemp -d)
cat > "$W/jpegfirst.c" <<'C'
#include "iimage.h"
__attribute__((constructor)) static void jpegFirst(void) { iiInsertCheckFunction(iiJPEGCheck, 3); }
C
gcc -shared -fPIC -O1 -I$R/include "$W/jpegfirst.c" -L$R/buildlib -liimod -o "$W/jpegfirst.so"
rm -rf "$F/golden"; mkdir -p "$F/golden"
sed "${FULL:+s/^#full\t//;}/^#/d" "$F/cases.tsv" | while IFS=$'\t' read -r name prog args check; do
  [ -z "$name" ] && continue
  case $prog in header|newstack) P=$R/flib/image/$prog;; clip) P=$R/clip/clip;; esac
  S=$(mktemp -d); cp "$F"/*.jpg "$S"; cd "$S"
  set +e
  LD_PRELOAD="$W/jpegfirst.so" $P $args < /dev/null | cat > "$F/golden/$name.stdout"
  echo "${PIPESTATUS[0]}" > "$F/golden/$name.rc"
  set -e
  [ -f o.mrc ] && cp o.mrc "$F/golden/$name.o.mrc"
  cd /; rm -rf "$S"
done
rm -rf "$W"
