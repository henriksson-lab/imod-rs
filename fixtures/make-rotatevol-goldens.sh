#!/bin/bash
# Regenerate fixtures/rotatevol/golden/ from the native reference rotatevol (see
# make-small-prog-goldens.sh for the layout).  Inputs: make-etomo-prog-inputs.sh.
exec "$(dirname "$0")/make-small-prog-goldens.sh" rotatevol flib/image/rotatevol
