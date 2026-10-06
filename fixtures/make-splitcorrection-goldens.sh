#!/bin/bash
# Regenerate fixtures/splitcorrection/golden from the native Python splitcorrection (IMOD/pysrc/splitcorrection) driving the
# native reference programs; see make-pyscript-goldens.py.  REF defaults to
# /tmp/imod-reference-build.
set -e
exec python3 "$(cd "$(dirname "$0")" && pwd)/make-pyscript-goldens.py" splitcorrection
