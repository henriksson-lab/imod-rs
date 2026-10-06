#!/bin/bash
# Regenerate fixtures/finishjoin/golden from the native Python finishjoin (IMOD/pysrc/finishjoin) driving the
# native reference programs; see make-pyscript-goldens.py.  REF defaults to
# /tmp/imod-reference-build.
set -e
exec python3 "$(cd "$(dirname "$0")" && pwd)/make-pyscript-goldens.py" finishjoin
