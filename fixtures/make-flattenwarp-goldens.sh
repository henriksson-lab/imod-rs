#!/bin/bash
# Regenerate fixtures/flattenwarp/golden/ from the native reference flattenwarp (see
# make-small-prog-goldens.sh for the layout).  Inputs: make-etomo-prog-inputs.sh.
exec "$(dirname "$0")/make-small-prog-goldens.sh" flattenwarp imodutil/flattenwarp
