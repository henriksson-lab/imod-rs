#!/bin/bash
# Inputs for the imodextract golden suite.  six.mod: native wmod2imod of a
# seeded WIMP file (10 contours in 6 objects); vsix.mod: native imodjoin of
# fixtures/model-view-clip-label.mod (a model with views) and six.mod, so the
# object views are shifted too; grp.mod: vsix.mod with four object groups
# (OGRP chunks, objgroup.c:193 layout) inserted before IEOF -- "first" {0,2},
# "second" {1,2,5}, "empty" {}, "last" {6,3}.  Run once; outputs committed.
set -e
R=${IMOD_REF:-/tmp/imod-reference-build}
export LD_LIBRARY_PATH=$R/buildlib
F=$(cd "$(dirname "$0")" && pwd)
S=$(mktemp -d)
cd "$S"
python3 - <<'PY'
objs = []
for k in range(10):
    pts = [(10 + k * 3 + i * 2.0, 20 + k * 1.5 + i, float(k % 4)) for i in range(3 + k % 3)]
    objs.append((240 + k % 6, pts))
out = [" Model file name........................six.wimp",
       " max # of object.......................   18",
       " # of node.............................  %4d" % sum(len(p) + 1 for _, p in objs),
       " # of object...........................    %d" % len(objs),
       "  Object sequence : "]
node = 1
for i, (c, pts) in enumerate(objs):
    out += ["  Object #:           %d" % (i + 1), " # of point:          %d" % len(pts),
            " Display switch:1  %d" % c, "     #    X       Y       Z      Mark    Label "]
    for x, y, z in pts:
        node += 1
        out.append("    %3d %7.2f %7.2f %7.2f   0" % (node, x, y, z))
    node += 1
out.append("  END")
open("six.wimp", "w").write("\n".join(out) + "\n")
PY
$R/imodutil/wmod2imod six.wimp six.mod
$R/imodutil/imodjoin "$F/model-view-clip-label.mod" six.mod vsix.mod > /dev/null
python3 - <<'PY'
import struct
d = open("vsix.mod", "rb").read()
i = d.rindex(b"IEOF")
def grp(name, objs):
    return (b"OGRP" + struct.pack(">i", 32 + 4 * len(objs)) + name.ljust(32, b"\0")
            + b"".join(struct.pack(">i", o) for o in objs))
g = grp(b"first", [0, 2]) + grp(b"second", [1, 2, 5]) + grp(b"empty", []) + grp(b"last", [6, 3])
open("grp.mod", "wb").write(d[:i] + g + d[i:])
PY
cp six.mod vsix.mod grp.mod "$F/imodextract/"
rm -rf "$S"
