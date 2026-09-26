#!/bin/bash
# Regenerate fixtures/tomopieces/golden/ from the native reference tomopieces (see
# make-small-prog-goldens.sh for the layout).
exec "$(dirname "$0")/make-small-prog-goldens.sh" tomopieces flib/image/tomopieces
