#!/bin/bash
# Regenerate fixtures/matchvol/golden/ from the native reference matchvol (see
# make-small-prog-goldens.sh for the layout).
exec "$(dirname "$0")/make-small-prog-goldens.sh" matchvol flib/image/matchvol
