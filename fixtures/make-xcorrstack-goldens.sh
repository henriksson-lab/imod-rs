#!/bin/bash
# Regenerate fixtures/xcorrstack/golden/ from the native reference xcorrstack (see
# make-small-prog-goldens.sh for the layout).
exec "$(dirname "$0")/make-small-prog-goldens.sh" xcorrstack flib/image/xcorrstack
