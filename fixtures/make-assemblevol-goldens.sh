#!/bin/bash
# Regenerate fixtures/assemblevol/golden/ from the native reference assemblevol (see
# make-small-prog-goldens.sh for the layout).
exec "$(dirname "$0")/make-small-prog-goldens.sh" assemblevol flib/image/assemblevol
