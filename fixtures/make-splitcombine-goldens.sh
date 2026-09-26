#!/bin/bash
# Regenerate fixtures/splitcombine/golden from the native Python splitcombine (IMOD/pysrc/splitcombine)
# driving the native reference programs; see make-pysetup-goldens.py for the
# case format and what is recorded.  REF defaults to /tmp/imod-reference-build.
set -e
F=$(cd "$(dirname "$0")" && pwd)
python3 "$F/make-pysetup-goldens.py" splitcombine
# Upstream bug fixed in translation, applied to the golden (BUGS.md, dual-axis
# combine scripts): the finishing file takes the command file's extension.
G=$F/splitcombine/golden
mv "$G/lowrad_pcm/volcombine-finish.com" "$G/lowrad_pcm/volcombine-finish.pcm"
sed -i 's/^volcombine-finish.com$/volcombine-finish.pcm/' "$G/lowrad_pcm.files"
