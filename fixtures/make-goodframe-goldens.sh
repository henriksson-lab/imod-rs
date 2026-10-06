#!/bin/bash
# Regenerate fixtures/goodframe/golden/ from the native reference goodframe (see
# make-small-prog-goldens.sh for the layout).  Inputs: make-etomo-prog-inputs.sh.
exec "$(dirname "$0")/make-small-prog-goldens.sh" goodframe flib/image/goodframe
