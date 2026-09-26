#!/bin/bash
# Regenerate fixtures/tomopitch/golden/ from the native reference tomopitch (see
# make-small-prog-goldens.sh for the layout).
exec "$(dirname "$0")/make-small-prog-goldens.sh" tomopitch flib/model/tomopitch
