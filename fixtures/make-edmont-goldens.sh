#!/bin/bash
# Regenerate fixtures/edmont/golden/ from the native reference edmont (see
# make-small-prog-goldens.sh for the layout).  Inputs: make-edmont-inputs.sh.
exec "$(dirname "$0")/make-small-prog-goldens.sh" edmont flib/model/edmont
