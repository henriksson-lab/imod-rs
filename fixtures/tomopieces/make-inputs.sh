#!/bin/bash
# Regenerates the tomopieces input (a small byte volume, whose header gives
# the size) with the native raw2mrc.
set -e
cd "$(dirname "$0")"
R=${IMOD_REF:-/tmp/imod-reference-build}
python3 -c "import numpy as np; np.arange(10*6*8,dtype=np.uint8).tofile('t.raw')"
rm -f tomo.mrc  # else raw2mrc leaves a tomo.mrc~ backup
LD_LIBRARY_PATH=$R/buildlib $R/mrc/raw2mrc -x 10 -y 6 -z 8 -t byte t.raw tomo.mrc >/dev/null
rm -f t.raw
