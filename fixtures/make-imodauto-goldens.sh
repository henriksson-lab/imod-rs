#!/bin/bash
# Regenerate fixtures/imodauto/golden/ from the native reference imodauto (see
# make-small-prog-goldens.sh for the layout).  Inputs: make-etomo-prog-inputs.sh.
export OMP_NUM_THREADS=1
exec "$(dirname "$0")/make-small-prog-goldens.sh" imodauto imodutil/imodauto
