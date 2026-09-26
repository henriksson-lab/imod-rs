#!/bin/bash
# Regenerate fixtures/solvematch/golden/ from the native reference solvematch (see
# make-small-prog-goldens.sh for the layout).
exec "$(dirname "$0")/make-small-prog-goldens.sh" solvematch flib/model/solvematch
