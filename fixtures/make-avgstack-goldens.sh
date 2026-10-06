#!/bin/bash
# Regenerate fixtures/avgstack/golden/ from the native reference avgstack (see
# make-small-prog-goldens.sh for the layout).  Inputs: make-etomo-prog-inputs.sh.
exec "$(dirname "$0")/make-small-prog-goldens.sh" avgstack flib/image/avgstack
