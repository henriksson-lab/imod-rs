#!/bin/bash
# Regenerate fixtures/alignframes/golden/ from the native reference alignframes
# (see make-small-prog-goldens.sh for the layout).  Inputs:
# make-alignframes-inputs.sh.  Rows with a fourth column run that native-
# equivalent input (an upstream bug fixed in the translation, BUGS.md); a fifth
# column is a sed expression applied to native's standard output.
export OMP_NUM_THREADS=1
exec "$(dirname "$0")/make-small-prog-goldens.sh" alignframes mrc/alignframes
