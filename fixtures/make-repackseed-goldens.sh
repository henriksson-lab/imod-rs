#!/bin/bash
# Regenerate fixtures/repackseed/golden/ from the native reference repackseed (see
# make-small-prog-goldens.sh for the layout).  Inputs: make-etomo-prog-inputs.sh.
export OMP_NUM_THREADS=1
exec "$(dirname "$0")/make-small-prog-goldens.sh" repackseed flib/model/repackseed
