#!/bin/bash
# Regenerate fixtures/alttomosetup/golden from the native Python alttomosetup (IMOD/pysrc/alttomosetup) driving the
# native reference programs; see make-pyscript-goldens.py.  REF defaults to
# /tmp/imod-reference-build.
set -e
exec python3 "$(cd "$(dirname "$0")" && pwd)/make-pyscript-goldens.py" alttomosetup
