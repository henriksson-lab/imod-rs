#!/bin/bash
# Regenerate fixtures/model2point/golden/ from the native reference model2point (see
# make-small-prog-goldens.sh for the layout).  Inputs: make-etomo-prog-inputs.sh.
export OMP_NUM_THREADS=1
exec "$(dirname "$0")/make-small-prog-goldens.sh" model2point flib/model/model2point
