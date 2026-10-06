#!/bin/bash
# Regenerate fixtures/boxstartend/golden/ from the native reference boxstartend (see
# make-small-prog-goldens.sh for the layout).  Inputs: make-etomo-prog-inputs.sh.
export OMP_NUM_THREADS=1
exec "$(dirname "$0")/make-small-prog-goldens.sh" boxstartend flib/model/boxstartend
