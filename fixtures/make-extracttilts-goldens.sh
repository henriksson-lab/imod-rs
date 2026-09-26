#!/bin/bash
# Regenerate fixtures/extracttilts/golden/ from the native reference extracttilts (see
# make-small-prog-goldens.sh for the layout).
#
# Exception (BUGS.md, extracttilts, fixed in translation): golden/et_interactive*
# (not et_interactive_eof) were written by the translation (`imod extracttilts`);
# native reads an uninitialised ifInfoFile there.  Restore them after running this.
exec "$(dirname "$0")/make-small-prog-goldens.sh" extracttilts flib/image/extracttilts
