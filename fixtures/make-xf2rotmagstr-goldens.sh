#!/bin/bash
# Regenerate fixtures/xf2rotmagstr/golden/ from the native reference xf2rotmagstr (see
# make-small-prog-goldens.sh for the layout).
exec "$(dirname "$0")/make-small-prog-goldens.sh" xf2rotmagstr flib/distort/xf2rotmagstr
