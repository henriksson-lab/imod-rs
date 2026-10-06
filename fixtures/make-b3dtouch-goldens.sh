#!/bin/bash
# Regenerate fixtures/b3dtouch/golden from the native Python b3dtouch (IMOD/pysrc/b3dtouch) driving the
# native reference programs; see make-pyscript-goldens.py.  REF defaults to
# /tmp/imod-reference-build.
set -e
exec python3 "$(cd "$(dirname "$0")" && pwd)/make-pyscript-goldens.py" b3dtouch
