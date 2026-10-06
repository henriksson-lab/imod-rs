#!/bin/bash
# Regenerate fixtures/imodextract/golden/ from the native reference imodextract
# (see make-small-prog-goldens.sh for the layout).  Inputs:
# make-imodextract-inputs.sh.
exec "$(dirname "$0")/make-small-prog-goldens.sh" imodextract imodutil/imodextract
