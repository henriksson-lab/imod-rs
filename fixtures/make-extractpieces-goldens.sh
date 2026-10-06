#!/bin/bash
# Regenerate fixtures/extractpieces/golden/ from the native reference extractpieces (see
# make-small-prog-goldens.sh for the layout).  Inputs: make-etomo-prog-inputs.sh.
exec "$(dirname "$0")/make-small-prog-goldens.sh" extractpieces flib/image/extractpieces
