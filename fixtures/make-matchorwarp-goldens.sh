#!/bin/bash
# Regenerate fixtures/matchorwarp/golden from the native Python matchorwarp (IMOD/pysrc/matchorwarp)
# driving the native reference programs; see make-pysetup-goldens.py for the
# case format and what is recorded.  REF defaults to /tmp/imod-reference-build.
set -e
F=$(cd "$(dirname "$0")" && pwd)
python3 "$F/make-pysetup-goldens.py" matchorwarp
# Upstream bugs fixed in translation, applied to the goldens (BUGS.md,
# dual-axis combine scripts): the "cannot find patch output file" error is a
# TypeError natively (nothing on stdout), and the "file to align" error names
# the patch file where the command file is meant.
G=$F/matchorwarp/golden
printf 'ERROR: matchorwarp - Cannot find name of patch output file in pc.com\n' > "$G/err_iter_noout.out"
sed -i 's/ in patch.out$/ in pc.com/' "$G/err_iter_nomat.out"
