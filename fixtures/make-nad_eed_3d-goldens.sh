#!/bin/bash
# Regenerate fixtures/nad_eed_3d/golden/ from the native reference nad_eed_3d (see
# make-small-prog-goldens.sh for the layout).  Inputs: make-etomo-prog-inputs.sh.
export OMP_NUM_THREADS=1
exec "$(dirname "$0")/make-small-prog-goldens.sh" nad_eed_3d mrc/nad_eed_3d
