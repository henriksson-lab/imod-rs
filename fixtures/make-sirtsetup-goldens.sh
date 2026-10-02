#!/bin/bash
# Regenerate fixtures/sirtsetup/golden from the native Python sirtsetup (IMOD/pysrc/sirtsetup)
# driving the native reference programs (and the native Python splittilt); see
# make-pysetup-goldens.py for the case format and what is recorded.  REF defaults
# to /tmp/imod-reference-build.
set -e
exec python3 "$(cd "$(dirname "$0")" && pwd)/make-pysetup-goldens.py" sirtsetup
