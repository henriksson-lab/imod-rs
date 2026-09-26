#!/bin/bash
# Regenerate fixtures/refinematch/golden/ from the native reference refinematch (see
# make-small-prog-goldens.sh for the layout).  The patch files were written by
# fixtures/findwarp/make-patches.py; the region models by the native point2model.
# Every input but p13.patch is fixtures/findwarp's (not stored twice).
SHARED="../findwarp/badinit2.xf ../findwarp/init.xf ../findwarp/p1.patch ../findwarp/p2.patch ../findwarp/p3.patch ../findwarp/p5.patch ../findwarp/p9.patch ../findwarp/reg1.mod ../findwarp/reg2.mod" \
  exec "$(dirname "$0")/make-small-prog-goldens.sh" refinematch flib/model/refinematch
