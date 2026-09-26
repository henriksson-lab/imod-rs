#!/bin/bash
# Regenerate fixtures/fixboundaries/golden/ from the native reference fixboundaries (see
# make-small-prog-goldens.sh for the layout).
exec "$(dirname "$0")/make-small-prog-goldens.sh" fixboundaries flib/image/fixboundaries
