#!/usr/bin/env python3
"""Writes the generated .mdoc inputs of fixtures/extracttilts (run in that
directory).  plain_mdoc.st.mdoc is hand-written and kept as is."""
def mdoc(name, n, fields, glob="PixelSpacing = 3.2\nImageFile = x.st\nImageSize = 4 4\n", titles=(), montage=False, zname="ZValue", skip=()):
    s = glob
    if montage: s += "Montage = 1\n"
    s += "\n"
    for t in titles: s += f"[T = {t}]\n\n"
    for z in range(n):
        s += f"[{zname} = {z}]\n"
        for k, f in fields.items():
            if (z, k) in skip: continue
            s += f"{k} = {f(z)}\n"
        s += "\n"
    open(name, "w").write(s)
T = lambda z: f"{-30 + 7.5*z:.2f}"
base = {"TiltAngle": T, "StagePosition": lambda z: f"{1.5+0.1*z:.3f} {-2.25+0.05*z:.3f}",
        "Magnification": lambda z: 26000 + 1000*(z % 2), "Intensity": lambda z: f"{0.123456+0.0001*z:.6f}",
        "ExposureDose": lambda z: f"{2.5+0.1*z:.3f}", "PixelSpacing": lambda z: "3.2",
        "Defocus": lambda z: f"{-2.345-0.01*z:.4f}", "ExposureTime": lambda z: "0.8",
        "Note": lambda z: "note " * (z + 1)}
order = [3, 0, 4, 1, 2]
dt = dict(base); dt["DateTime"] = lambda z: f"21-Jan-20  14:{22+order[z]:02d}:{10*order[z]:02d}"
mdoc("plain.st.mdoc", 5, dt, titles=["SerialEM: Digitized", "    Tilt axis angle = 85.3, binning = 1  spot = 8  camera = 0 bidir = -9.0"])
mdoc("plain_bidir.st.mdoc", 5, base)
mdoc("seri_all.st.mdoc", 17, base)
mdoc("seri_tilt.st.mdoc", 5, base)
mont = dict(base); mont["PieceCoordinates"] = lambda z: f"{100*(z%2)} 0 {z//2}"
mdoc("plain6.st.mdoc", 6, mont, montage=True)
mdoc("other_missing.mdoc", 5, base, skip={(2, "Defocus"), (3, "TiltAngle")})
open("series.mdoc", "w").write("ImageSeries = 1\n\n[Image = a.tif]\nTiltAngle = 3\n\n[Image = b.tif]\nTiltAngle = 4\n")
dp = dict(dt); mdoc("other_partial_dt.mdoc", 5, dp, skip={(1, "DateTime")})
bm = dict(base); bm["DateTime"] = lambda z: f"21-Foo-20  14:22:{z:02d}"
mdoc("other_badmonth.mdoc", 5, bm)
tie = dict(base); tie["DateTime"] = lambda z: "21-Jan-20  14:22:00" if z < 3 else f"21-Jan-20  14:21:{z:02d}"
mdoc("other_ties.mdoc", 5, tie)
zd = dict(dt); zd["ExposureDose"] = lambda z: "0" if z == 2 else "2.0"
mdoc("other_zerodose.mdoc", 5, zd)
mdoc("other_dosym.mdoc", 5, base, titles=["    Tilt axis angle = 85.3, dosym = 7.4 spot=1"])
mdoc("other_badangle.mdoc", 5, base, titles=["Tilt axis angle = 85.3, bidir = x7.4"])
mdoc("other_noeq.mdoc", 5, base, titles=["Tilt axis angle = 85.3, bidir"])
