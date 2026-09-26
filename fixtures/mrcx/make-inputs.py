#!/usr/bin/env python3
"""Author the mrcx/mrcinfo fixture inputs: small MRC files of every mode
mrcx handles, in both byte orders, with new- and old-style headers,
extended headers, trailing bytes and truncated data.  Deterministic.

usage: make-inputs.py OUTDIR
"""
import struct, sys, os
import numpy as np

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(1234)


def header(e, nx, ny, nz, mode, amin, amax, amean, next_=0, old=False,
           labels=(b"mrcx fixture",), idtype=0, ispg=0, tilt=(0,) * 6,
           org=(0.0, 0.0, 0.0)):
    h = bytearray(1024)
    struct.pack_into(e + "10i", h, 0, nx, ny, nz, mode, 0, 0, 0, nx, ny, nz)
    struct.pack_into(e + "6f", h, 40, nx * 1.5, ny * 1.5, nz * 1.5, 90, 90, 90)
    struct.pack_into(e + "3i", h, 64, 1, 2, 3)
    struct.pack_into(e + "3f", h, 76, amin, amax, amean)
    struct.pack_into(e + "2i", h, 88, ispg, next_)
    struct.pack_into(e + "h", h, 96, 1000)
    h[104:108] = b"SERI"
    struct.pack_into(e + "i", h, 108, 20140)
    struct.pack_into(e + "4h", h, 128, 0, 32, 0, 0)
    struct.pack_into(e + "4f", h, 136, 0.5, 2.5, -1e-5, 1e-5)
    struct.pack_into(e + "2i", h, 152, 1146047817, 1)
    struct.pack_into(e + "6h", h, 160, idtype, 3, 7, -9, 1234, -5)
    struct.pack_into(e + "6f", h, 172, *tilt)
    if old:
        struct.pack_into(e + "6h", h, 196, 5, 6, 7, 8, 9, 10)
        struct.pack_into(e + "3f", h, 208, *org)
    else:
        struct.pack_into(e + "3f", h, 196, *org)
        h[208:212] = b"MAP "
        h[212:216] = bytes([0x44 if e == "<" else 0x11, 0x44 if e == "<" else 0x11, 0, 0])
        struct.pack_into(e + "f", h, 216, 3.25)
    struct.pack_into(e + "i", h, 220, len(labels))
    for i, lab in enumerate(labels):
        h[224 + 80 * i:224 + 80 * (i + 1)] = lab.ljust(80)
    return bytes(h)


def write(name, e, arr, mode, **kw):
    nz, ny, nx = arr.shape[:3]
    if mode in (3, 4):
        nx = arr.shape[2] // 2
    data = arr.astype(arr.dtype.newbyteorder(e)).tobytes()
    a = arr.astype(np.float64)
    h = header(e, nx, ny, nz, mode, float(a.min()), float(a.max()),
               float(a.mean()), **{k: v for k, v in kw.items()
                                   if k not in ("ext", "trail", "cut")})
    ext = kw.get("ext", b"")
    body = h + ext + data + kw.get("trail", b"")
    if kw.get("cut"):
        body = body[:-kw["cut"]]
    open(os.path.join(out, name), "wb").write(body)


for e, tag in (("<", "le"), (">", "be")):
    write(f"byte_{tag}.mrc", e, rng.integers(0, 256, (2, 6, 8), dtype=np.uint8), 0)
    write(f"short_{tag}.mrc", e, rng.integers(-3000, 3000, (2, 6, 8), dtype=np.int16), 1,
          idtype=2, ispg=1)
    write(f"float_{tag}.mrc", e, rng.normal(0, 1e-3, (3, 5, 7)).astype(np.float32), 2,
          tilt=(1.5, -2.25, 0, 1e-7, 3e8, -0.0), org=(12.5, -3.75, 1e20))
    write(f"cshort_{tag}.mrc", e, rng.integers(-99, 99, (2, 3, 8), dtype=np.int16), 3)
    write(f"cfloat_{tag}.mrc", e, rng.normal(0, 100, (2, 3, 8)).astype(np.float32), 4)
    write(f"rgb_{tag}.mrc", e, rng.integers(0, 256, (2, 3, 4, 3), dtype=np.uint8), 16)
    write(f"ext_odd_{tag}.mrc", e, rng.normal(5, 2, (1, 4, 6)).astype(np.float32), 2,
          next_=7, ext=bytes(range(1, 8)))
    write(f"old_{tag}.mrc", e, rng.integers(-500, 500, (2, 4, 5), dtype=np.int16), 1,
          old=True, org=(1.0, 2.0, 3.0))
    write(f"trail100_{tag}.mrc", e, rng.normal(0, 1, (1, 4, 4)).astype(np.float32), 2,
          trail=b"T" * 100)
    write(f"trail600_{tag}.mrc", e, rng.normal(0, 1, (1, 4, 4)).astype(np.float32), 2,
          trail=b"U" * 600)
    write(f"cut_{tag}.mrc", e, rng.normal(0, 1, (2, 4, 4)).astype(np.float32), 2, cut=40)
    write(f"cutbyte_{tag}.mrc", e, rng.integers(0, 256, (2, 4, 4), dtype=np.uint8), 0, cut=5)
    write(f"half_{tag}.mrc", e, rng.normal(0, 1, (1, 2, 2)).astype(np.float16), 12)
    write(f"labels_{tag}.mrc", e, np.zeros((1, 2, 2), np.float32), 2,
          labels=tuple(b"label %d" % i for i in range(10)))

# Not MRC at all, and too short to hold a header record.
open(os.path.join(out, "notmrc.bin"), "wb").write(bytes(range(256)) * 5)
open(os.path.join(out, "tiny.bin"), "wb").write(b"0123456789")
# mrcinfo: extreme and tiny statistics, many labels claimed.
h = bytearray(header("<", 2, 2, 1, 2, -3.4e38, 1e-30, 1.17549435e-38))
struct.pack_into("<i", h, 220, 11)
open(os.path.join(out, "extreme_le.mrc"), "wb").write(bytes(h) + bytes(16))
