#!/bin/bash
# Regenerate fixtures/imod2patch/golden/ from the native reference imod2patch (see
# make-small-prog-goldens.sh for the layout).  Inputs: make-etomo-prog-inputs.sh.
exec "$(dirname "$0")/make-small-prog-goldens.sh" imod2patch imodutil/imod2patch
