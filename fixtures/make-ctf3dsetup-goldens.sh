#!/bin/bash
# Regenerate fixtures/ctf3dsetup/golden from the native Python ctf3dsetup (IMOD/pysrc/ctf3dsetup) driving the
# native reference programs; see make-pyscript-goldens.py.  REF defaults to
# /tmp/imod-reference-build.
set -e
exec python3 "$(cd "$(dirname "$0")" && pwd)/make-pyscript-goldens.py" ctf3dsetup
