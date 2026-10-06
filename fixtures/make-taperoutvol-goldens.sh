#!/bin/bash
# Regenerate fixtures/taperoutvol/golden/ from the native reference taperoutvol (see
# make-small-prog-goldens.sh for the layout).  Inputs: make-etomo-prog-inputs.sh.
exec "$(dirname "$0")/make-small-prog-goldens.sh" taperoutvol flib/image/taperoutvol
