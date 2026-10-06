#!/bin/bash
# Regenerate fixtures/subimage/golden/ from the native reference subimage (see
# make-small-prog-goldens.sh for the layout).  Inputs: make-etomo-prog-inputs.sh.
export OMP_NUM_THREADS=1
exec "$(dirname "$0")/make-small-prog-goldens.sh" subimage flib/image/subimage
