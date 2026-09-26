#!/usr/bin/env python3
"""Author the tifinfo fixture inputs.  Deterministic.

tifinfo.c was written for a big-endian host: it swaps whenever the order
word is not "MM".  On a little-endian host that makes a real TIFF of either
order unreadable (BUGS.md), so besides the real files this writes the two
"cross-labelled" files the native binary can walk (and the translation,
which reads real TIFFs on any host -- BUGS.md, fixed in translation -- treats
as the corrupt files they are): an "II" file whose body
is big-endian (swapped into sense) and an "MM" file whose body is
little-endian (read as is, with the short-value >> 16 rule applied).

usage: make-inputs.py OUTDIR
"""
import os, sys
import numpy as np
import tifffile

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
a = (np.arange(24 * 20) % 251).astype(np.uint8).reshape(20, 24)


def w(name, data, order, relabel=None, **kw):
    p = os.path.join(out, name)
    tifffile.imwrite(p, data, byteorder=order, software="fixture", **kw)
    if relabel:
        b = bytearray(open(p, "rb").read())
        b[0:2] = relabel
        open(p, "wb").write(bytes(b))


for tag, order, relabel in (("iibe", ">", b"II"), ("mmle", "<", b"MM")):
    w(f"{tag}_byte.tif", a, order, relabel, description="byte page")
    w(f"{tag}_multi.tif", np.stack([a, a + 1, a + 2]).astype(np.uint16), order,
      relabel, description="three pages")
    w(f"{tag}_tiled.tif", np.stack([a, a]).astype(np.float32)[:, :16, :16],
      order, relabel, tile=(16, 16), description="tiled float")
# Real files: native loops forever on MM and segfaults on II (BUGS.md); the
# translation reads them (fixed in translation).
w("real_be.tif", a, ">")
w("real_le.tif", a, "<")
# Real counterparts of the cross-labelled files: relabelling restores the
# order word that matches the body.
for kind in ("byte", "multi", "tiled"):
    for src, dst, label in (("mmle", "real_le", b"II"), ("iibe", "real_be", b"MM")):
        b = bytearray(open(os.path.join(out, f"{src}_{kind}.tif"), "rb").read())
        b[0:2] = label
        open(os.path.join(out, f"{dst}_{kind}.tif"), "wb").write(bytes(b))
open(os.path.join(out, "nottif.bin"), "wb").write(b"XY*\0" + bytes(20))
open(os.path.join(out, "badver.tif"), "wb").write(b"II\x2b\x00\x08\x00\x00\x00" + bytes(8))
open(os.path.join(out, "empty.tif"), "wb").write(b"")
# Header only, IFD offset 0: prints the banner and stops.
open(os.path.join(out, "noifd.tif"), "wb").write(b"MM\x2a\x00\x00\x00\x00\x00")
