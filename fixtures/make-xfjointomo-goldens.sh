#!/bin/bash
# Regenerate fixtures/xfjointomo/golden/ from the native reference xfjointomo (see
# make-small-prog-goldens.sh for the layout).  Inputs: make-etomo-prog-inputs.sh.
exec "$(dirname "$0")/make-small-prog-goldens.sh" xfjointomo flib/model/xfjointomo
