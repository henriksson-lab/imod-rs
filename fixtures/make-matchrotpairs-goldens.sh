#!/bin/bash
# Regenerate fixtures/matchrotpairs/{inputs,golden} from the native Python matchrotpairs
# (IMOD/pysrc/matchrotpairs, with IMOD/pysrc/tiltmatch.py) driving the native reference
# newstack, tiltxcorr, xfsimplex and xfproduct; see make-pysetup-goldens.py for the case
# format and what is recorded.  REF defaults to /tmp/imod-reference-build.
set -e
exec python3 "$(cd "$(dirname "$0")" && pwd)/make-pysetup-goldens.py" matchrotpairs
