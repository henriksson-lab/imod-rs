#!/bin/bash
# Regenerate fixtures/remapmodel/golden/ from the native reference remapmodel (see
# make-small-prog-goldens.sh for the layout).  Inputs: make-etomo-prog-inputs.sh.
exec "$(dirname "$0")/make-small-prog-goldens.sh" remapmodel flib/model/remapmodel
