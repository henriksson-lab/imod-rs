#!/bin/bash
# Regenerate fixtures/setupcombine/golden from the native Python setupcombine (IMOD/pysrc/setupcombine)
# driving the native reference programs; see make-pysetup-goldens.py for the
# case format and what is recorded.  REF defaults to /tmp/imod-reference-build.
set -e
F=$(cd "$(dirname "$0")" && pwd)
python3 "$F/make-pysetup-goldens.py" setupcombine
# Upstream bug fixed in translation, applied to the golden (BUGS.md, dual-axis
# combine scripts): the missing-temporary-directory message lacks a space.
sed -i 's/^\(ERROR: setupcombine - nodir\)does not exist/\1 does not exist/' \
    "$F/setupcombine/golden/err_tempdir.out"
