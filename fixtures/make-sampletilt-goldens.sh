#!/bin/bash
# Regenerate fixtures/sampletilt/golden from the native Python sampletilt (IMOD/pysrc/sampletilt) driving the
# native reference programs; see make-pyscript-goldens.py.  REF defaults to
# /tmp/imod-reference-build.
set -e
exec python3 "$(cd "$(dirname "$0")" && pwd)/make-pyscript-goldens.py" sampletilt
