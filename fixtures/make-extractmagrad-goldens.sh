#!/bin/bash
# Regenerate fixtures/extractmagrad/golden/ from the native reference extractmagrad (see
# make-small-prog-goldens.sh for the layout).
exec "$(dirname "$0")/make-small-prog-goldens.sh" extractmagrad flib/image/extractmagrad
