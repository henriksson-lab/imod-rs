#!/bin/bash
# Regenerate fixtures/dualvolmatch/golden from the native Python dualvolmatch (IMOD/pysrc/dualvolmatch)
# driving the native reference programs; see make-pysetup-goldens.py for the
# case format and what is recorded.  REF defaults to /tmp/imod-reference-build.
set -e
exec python3 "$(cd "$(dirname "$0")" && pwd)/make-pysetup-goldens.py" dualvolmatch
