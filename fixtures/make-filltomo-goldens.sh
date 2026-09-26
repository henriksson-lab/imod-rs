#!/bin/bash
# Regenerate fixtures/filltomo/golden/ from the native reference filltomo (see
# make-small-prog-goldens.sh for the layout).
exec "$(dirname "$0")/make-small-prog-goldens.sh" filltomo flib/model/filltomo
