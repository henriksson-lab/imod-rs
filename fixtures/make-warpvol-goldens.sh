#!/bin/bash
# Regenerate fixtures/warpvol/golden/ from the native reference warpvol (see
# make-small-prog-goldens.sh for the layout).
# vf.mrc and vs.mrc are fixtures/matchvol's (the same seeded volumes
# both suites use).
SHARED="../matchvol/vf.mrc ../matchvol/vs.mrc" exec "$(dirname "$0")/make-small-prog-goldens.sh" warpvol flib/image/warpvol
