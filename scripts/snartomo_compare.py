# snartomo_compare.py NATDIR RSDIR FILE...  -> "VERDICT\tdetails"
#
# Output parity for scripts/bench_snartomo.sh, with the masking rules of
# tests/common and scripts/e2e_ts01.sh (E2E.md "Comparison rules"):
#   MRC   label date stamps and the uninitialised label slots past nlabl
#         masked (BUGS.md 2), then every data byte compared
#   model Imod.name past its NUL and the MINX chunk's oscale/orot masked
#         (heap residue in native, CLAUDE.md)
#   TIFF  DateTime masked (BUGS.md 7)
#   JPEG  byte compare; if the encoded bytes differ, the decoded pixels are
#         compared: PIXEL = same pixels, JPEG = lossy re-encoding differs
#         (the Qt/libjpeg encoder boundary is a Rust crate in imod-rs;
#         RUST_NATIVE_BACKENDS_PLAN.md), with the decoded max/mean difference
#   text  date stamps, .tmp.<pid> names, "Shell/Python PID" masked
# Verdicts, worst first: MISSING, DIFF, JPEG, PIXEL, IDENT.
import os, re, struct, sys
import numpy as np

nd, rd = sys.argv[1], sys.argv[2]
rank = {'IDENT': 0, 'PIXEL': 1, 'JPEG': 2, 'DIFF': 3, 'MISSING': 4}
jp = [0, 0, 0.0, 0]  # jpeg files re-encoded differently: count, max|d|, sum mean|d|, identical count
worst, out = 'IDENT', []

def bump(v):
    global worst
    if rank[v] > rank[worst]: worst = v

def mask_mrc(x):
    x = bytearray(x)
    nlabl = struct.unpack('<i', bytes(x[220:224]))[0]
    if 0 <= nlabl <= 10:
        for i in range(10):
            s = 224 + 80 * i
            if i < nlabl: x[s + 56:s + 76] = b' ' * 20
            else: x[s:s + 80] = b' ' * 80
    return bytes(x)

def mask_model(b):
    x = bytearray(b)
    nul = x.find(b'\0', 8, 136)
    if nul >= 0: x[nul:136] = b'\0' * (136 - nul)
    p = x.find(b'MINX\0\0\0\x48')
    if p >= 0:
        x[p + 8:p + 20] = b'\0' * 12
        x[p + 32:p + 44] = b'\0' * 12
    return bytes(x)

STAMP = re.compile(rb'\d\d-[A-Z][a-z][a-z]-\d\d +\d\d:\d\d:\d\d')
TMPPID = re.compile(rb'\.tmp\.\d+')
PID = re.compile(rb'(Shell|Python) PID: *\d+')
def text_mask(a):
    return PID.sub(b'PID', TMPPID.sub(b'.tmp.<pid>', STAMP.sub(b'<stamp>', a)))

def same_stream(pa, pb, off):
    with open(pa, 'rb') as A, open(pb, 'rb') as Bf:
        A.seek(off); Bf.seek(off)
        while True:
            x, y = A.read(1 << 24), Bf.read(1 << 24)
            if x != y: return False
            if not x: return True

def mrc_diff(pa, pb, ha):
    nx, ny, nz, mode = struct.unpack('<4i', ha[:16])
    nxt = struct.unpack('<i', ha[92:96])[0]
    dt = {0: np.uint8, 1: np.int16, 2: np.float32, 6: np.uint16}.get(mode)
    if dt is None: return 'data differs (mode %d)' % mode
    a = np.memmap(pa, dt, 'r', 1024 + nxt, (nx * ny * nz,))
    b = np.memmap(pb, dt, 'r', 1024 + nxt, (nx * ny * nz,))
    n, m = 0, 0.0
    for s in range(0, a.size, 1 << 24):
        x, y = a[s:s + (1 << 24)].astype(np.float64), b[s:s + (1 << 24)].astype(np.float64)
        d = np.abs(x - y); d[np.isnan(x) & np.isnan(y)] = 0
        n += int((d != 0).sum()); m = max(m, float(np.nanmax(d)) if d.size else 0)
    return '%d of %d voxels differ, max|d| %.4g' % (n, a.size, m)

for f in sys.argv[3:]:
    pn, pr = os.path.join(nd, f), os.path.join(rd, f)
    if not os.path.exists(pn) and not os.path.exists(pr):
        out.append('%s:neither-wrote' % f); bump('MISSING'); continue
    if not os.path.exists(pn): out.append('%s:native-missing' % f); bump('MISSING'); continue
    if not os.path.exists(pr): out.append('%s:MISSING' % f); bump('MISSING'); continue
    sa, sb = os.path.getsize(pn), os.path.getsize(pr)
    with open(pn, 'rb') as A, open(pr, 'rb') as Bf:
        ha, hb = A.read(1024), Bf.read(1024)
    if len(ha) == 1024 and ha[208:212] in (b'MAP ', b'MAP\x00'):
        if sa != sb:
            out.append('%s:DIFF(size %d vs %d)' % (f, sa, sb)); bump('DIFF'); continue
        mh = mask_mrc(ha) == mask_mrc(hb)
        md = same_stream(pn, pr, 1024)
        if mh and md: out.append('%s:IDENT' % f); continue
        why = [] if mh else ['header differs']
        if not md: why.append(mrc_diff(pn, pr, ha))
        out.append('%s:DIFF(%s)' % (f, '; '.join(why))); bump('DIFF'); continue
    a, b = open(pn, 'rb').read(), open(pr, 'rb').read()
    if a[:2] in (b'II', b'MM'):
        DT = re.compile(rb'\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}')
        if DT.sub(b'?' * 19, a) == DT.sub(b'?' * 19, b): out.append('%s:IDENT' % f); continue
        out.append('%s:DIFF' % f); bump('DIFF'); continue
    if a[:3] == b'\xff\xd8\xff':
        if a == b: out.append('%s:IDENT' % f); continue
        from PIL import Image
        import io
        ia, ib = np.asarray(Image.open(io.BytesIO(a))), np.asarray(Image.open(io.BytesIO(b)))
        if ia.shape != ib.shape:
            out.append('%s:DIFF(jpeg shape)' % f); bump('DIFF'); continue
        d = np.abs(ia.astype(int) - ib.astype(int))
        jp[0] += 1; jp[1] = max(jp[1], int(d.max())); jp[2] += float(d.mean())
        bump('PIXEL' if d.max() == 0 else 'JPEG'); continue
    if a[:4] == b'IMOD':
        if mask_model(a) == mask_model(b): out.append('%s:IDENT' % f); continue
        out.append('%s:DIFF(model)' % f); bump('DIFF'); continue
    if a == b or text_mask(a) == text_mask(b): out.append('%s:IDENT' % f); continue
    la, lb = text_mask(a).split(b'\n'), text_mask(b).split(b'\n')
    nl = sum(1 for x, y in zip(la, lb) if x != y) + abs(len(la) - len(lb))
    out.append('%s:DIFF(%d lines)' % (f, nl)); bump('DIFF')

if jp[0]:
    out.append('%d JPEGs encoded differently (decoded max|d| %d, mean|d| %.3f)'
               % (jp[0], jp[1], jp[2] / jp[0]))
out = [o for o in out if not o.endswith(':IDENT')] or ['all %d IDENT' % (len(sys.argv) - 3)]
print(worst + '\t' + ' '.join(out))
