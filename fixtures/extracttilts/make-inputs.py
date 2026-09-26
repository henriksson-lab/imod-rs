#!/usr/bin/env python3
"""Deterministic MRC files with SerialEM / Agard / FEI1 extended headers."""
import struct, sys, numpy as np


def header(nx, ny, nz, mode, nxt, ext_type=b"\0\0\0\0", nint=0, nreal=0,
           labels=(), delta=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), amin=0., amax=1., amean=0.5):
    h = bytearray(1024)
    struct.pack_into("<10i", h, 0, nx, ny, nz, mode, 0, 0, 0, nx, ny, nz)
    struct.pack_into("<6f", h, 40, nx * delta[0], ny * delta[1], nz * delta[2], 90., 90., 90.)
    struct.pack_into("<3i", h, 64, 1, 2, 3)
    struct.pack_into("<3f", h, 76, amin, amax, amean)
    struct.pack_into("<2i", h, 88, 0, nxt)
    h[104:108] = ext_type
    struct.pack_into("<i", h, 108, 20140)
    struct.pack_into("<2h", h, 128, nint, nreal)
    struct.pack_into("<3f", h, 196, *origin)
    h[208:212] = b"MAP "
    h[212:216] = bytes([0x44, 0x44, 0, 0])
    struct.pack_into("<i", h, 220, len(labels))
    for i, lab in enumerate(labels):
        b = lab.encode().ljust(80)[:80]
        h[224 + 80 * i:224 + 80 * (i + 1)] = b
    return h


def data(nx, ny, nz, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=nx * ny * nz, dtype=np.uint8).tobytes()


def write(name, h, ext, d):
    with open(name, "wb") as f:
        f.write(h); f.write(ext); f.write(d)


def seri(name, tilts, flags, extra=None, labels=(), pieces=None, nz=None, pad=0):
    """SerialEM: nint = bytes/section, nreal = flags."""
    sizes = [2, 6, 4, 2, 2, 4, 2, 4, 2, 4, 2]
    nbytes = sum(s for i, s in enumerate(sizes) if flags & (1 << i))
    nz = nz or len(tilts)
    ext = bytearray()
    for z in range(nz):
        sec = bytearray()
        for i, s in enumerate(sizes):
            if not flags & (1 << i):
                continue
            if i == 0:
                sec += struct.pack("<h", int(round(tilts[z] * 100)))
            elif i == 1:
                px, py, pz = pieces[z] if pieces else (0, 0, z)
                sec += struct.pack("<3H", px, py, pz)
            elif i == 2:
                sec += struct.pack("<2h", 25 * z - 40, 1000 - 13 * z)
            elif i == 3:
                sec += struct.pack("<h", 150 + z)
            elif i == 4:
                sec += struct.pack("<h", 12000 + 37 * z)
            elif i == 5:
                sec += struct.pack("<2h", 3000 + 7 * z, -(12 * 256) - 3)
            else:
                sec += bytes(s)
        ext += sec
    ext += bytes(pad)
    h = header(4, 4, nz, 0, len(ext), b"SERI", nbytes, flags, labels)
    write(name, h, bytes(ext), data(4, 4, nz))


def agard(name, tilts, nint=0, nreal=6, labels=()):
    ext = bytearray()
    for z, t in enumerate(tilts):
        ext += struct.pack("<%di" % nint, *range(nint)) if nint else b""
        vals = [t] + [0.5 * z + k for k in range(1, nreal)]
        ext += struct.pack("<%df" % nreal, *vals[:nreal])
    h = header(4, 4, len(tilts), 0, len(ext), b"AGAR", nint, nreal, labels)
    write(name, h, bytes(ext), data(4, 4, len(tilts)))


def fei(name, tilts, mask=(1 << 7) | (1 << 6) | 1, times=None, version=b"5.0", labels=(),
        blocksize=768, degree_bug=False):
    ext = bytearray()
    for z, t in enumerate(tilts):
        sec = bytearray(blocksize)
        struct.pack_into("<i", sec, 0, blocksize)
        struct.pack_into("<i", sec, 8, mask)
        tm = times[z] if times else 45000.0 + z * 0.001
        struct.pack_into("<d", sec, 12, tm)
        vs = b"4.4.0.4981" if degree_bug else version
        sec[68:68 + len(vs) + 1] = vs + b"\0"
        struct.pack_into("<d", sec, 92, (2.5 + 0.1 * z) * 1e20)
        struct.pack_into("<d", sec, 100, t * (0.017453292519943295 if degree_bug else 1.0))
        ext += sec
    h = header(4, 4, len(tilts), 0, len(ext), b"FEI1", 0, 0, labels)
    write(name, h, bytes(ext), data(4, 4, len(tilts)))


def plain(name, nz, labels=(), nx=4, ny=4):
    h = header(nx, ny, nz, 0, 0, b"\0\0\0\0", 0, 0, labels)
    write(name, h, b"", data(nx, ny, nz))


if __name__ == "__main__":
    tilts = [-60.0 + 7.5 * i for i in range(17)]
    lab_bidir = ["SerialEM: Acquired on Tecnai    ", "Tilt axis angle = 85.3, binning = 1  bidir = -20.0"]
    lab_dosym = ["Tilt axis angle = -5.1, binning = 2  dosym = 0.00  spot = 9"]
    seri("seri_tilt.st", tilts, 1)
    seri("seri_all.st", tilts, 1 | 4 | 8 | 16 | 32, labels=lab_bidir)
    seri("seri_mag.st", tilts, 8)
    seri("seri_mont.st", [10.0] * 6, 1 | 2 | 16,
         pieces=[(0, 0, 1), (100, 0, 1), (0, 0, 0), (100, 0, 0), (0, 100, 2), (100, 100, 2)])
    seri("seri_mont_notilt.st", [0.0] * 4, 2 | 16,
         pieces=[(0, 0, 0), (100, 0, 0), (0, 0, 1), (100, 0, 1)])
    seri("seri_zero.st", [0.0] * 5 + [0.05, 100.0], 1)
    seri("seri_dosym.st", tilts[:9], 1 | 16, labels=lab_dosym)
    agard("agard.st", tilts[:9])
    agard("agard_int.st", tilts[:5], nint=2, nreal=3, labels=lab_bidir)
    agard("agard_noflt.st", tilts[:5], nint=2, nreal=0)
    fei("fei.st", [-30.0, 0.0, 30.0, -15.0, 15.0], times=[45000.2, 45000.0, 45000.3, 45000.1, 45000.4])
    fei("fei_bug.st", [-30.0, 0.0, 30.0], degree_bug=True)
    fei("fei_notilt.st", [1.0, 2.0], mask=1)
    fei("fei_notime.st", [1.0, 2.0], mask=1 << 7)
    plain("plain.st", 5)
    plain("plain_bidir.st", 5, labels=lab_bidir)
    plain("plain_mdoc.st", 5, labels=lab_bidir)
