#!/bin/bash
# Regenerate fixtures/serieswatcher/golden from the native Python serieswatcher (IMOD/pysrc/serieswatcher) driving the
# native reference programs; see make-pyscript-goldens.py.  REF defaults to
# /tmp/imod-reference-build.
set -e
exec python3 "$(cd "$(dirname "$0")" && pwd)/make-pyscript-goldens.py" serieswatcher
