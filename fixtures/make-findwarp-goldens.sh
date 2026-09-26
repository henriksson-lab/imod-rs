#!/bin/bash
# Regenerate fixtures/findwarp/golden/ from the native reference findwarp (see
# make-small-prog-goldens.sh for the layout).  The patch files were written by
# fixtures/findwarp/make-patches.py; the region models by the native point2model.
exec "$(dirname "$0")/make-small-prog-goldens.sh" findwarp flib/model/findwarp
