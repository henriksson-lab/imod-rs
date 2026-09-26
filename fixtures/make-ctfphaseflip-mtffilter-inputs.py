#!/usr/bin/env python3
"""Writes the seeded synthetic inputs in fixtures/ctfphaseflip and fixtures/mtffilter.

The image stacks are written by the native reference `raw2mrc` (REF, default
/tmp/imod-reference-build) so the MRC container is IMOD's own; the text inputs
(tilt angles, defocus files in each format ctfutils.cpp reads, transforms, an
MTF curve, dose files and an mdoc) are written here.
"""
import os, subprocess, sys
import numpy as np

REF = os.environ.get('REF', '/tmp/imod-reference-build')
HERE = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ, AUTODOC_DIR=REF + '/autodoc', LD_LIBRARY_PATH=REF + '/buildlib')
rng = np.random.default_rng(20260926)


def stack(dirname, name, nx, ny, nz, kind):
    y, x = np.mgrid[0:ny, 0:nx]
    planes = []
    for _ in range(nz):
        img = np.zeros((ny, nx))
        for _ in range(8):
            cx, cy, r = rng.uniform(0, nx), rng.uniform(0, ny), rng.uniform(2, max(3, nx / 8))
            img += rng.uniform(-1, 1) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r * r))
        planes.append(img + 0.3 * rng.standard_normal((ny, nx)))
    a = np.array(planes)
    if kind == 'byte':
        a = np.clip(a * 40 + 128, 0, 255).astype(np.uint8)
    elif kind == 'short':
        a = np.clip(a * 800, -32000, 32000).astype(np.int16)
    else:
        a = (a * 3 + 10).astype(np.float32)
    raw = os.path.join(HERE, dirname, name + '.raw')
    a.tofile(raw)
    subprocess.run([REF + '/mrc/raw2mrc', '-x', str(nx), '-y', str(ny), '-z', str(nz), '-t', kind,
                    raw, os.path.join(HERE, dirname, name + '.mrc')], env=env, check=True,
                   stdout=subprocess.DEVNULL)
    os.remove(raw)


def write(dirname, name, lines):
    with open(os.path.join(HERE, dirname, name), 'w') as f:
        f.write('\n'.join(lines) + '\n')


C = 'ctfphaseflip'
stack(C, 'f48', 48, 40, 7, 'float')
stack(C, 'b45', 45, 38, 5, 'byte')
stack(C, 's160', 160, 32, 5, 'short')
a7 = np.linspace(-45, 45, 7)
a5 = np.linspace(-40, 40, 5)
write(C, 't7.tlt', ['%8.2f' % v for v in a7])
write(C, 't5.tlt', ['%8.2f' % v for v in a5])
write(C, 'v2.defocus', ['%d %d %.2f %.2f %d%s' % (i + 1, i + 1, a7[i], a7[i], 3000 + 60 * i,
                                                  ' 2' if i == 0 else '') for i in range(0, 7, 3)])
write(C, 'v1off.defocus', ['%d %d %.2f %.2f %d' % (i, i, a7[i], a7[i], 2500 + 50 * i)
                           for i in range(1, 7, 2)])
write(C, 'v5.defocus', ['%d %d %.2f %.2f %d%s' % (i + 1, i + 1, a5[i], a5[i], 1800 + 90 * i,
                                                  ' 2' if i == 0 else '') for i in range(5)])
write(C, 'single.defocus', ['4 4 0. 0. 3500 2'])
write(C, 'astig.defocus', ['1 0 0. 0. 0. 3'] + [
    '%d %d %.2f %.2f %d %d %.1f' % (i + 1, i + 1, a7[i], a7[i], 2800 + 40 * i, 3100 + 30 * i,
                                    -70 + 25 * i) for i in range(0, 7, 2)])
write(C, 'phasecut.defocus', ['36 0 0. 0. 0. 3'] + [
    '%d %d %.2f %.2f %d %.1f %.4f' % (i + 1, i + 1, a7[i], a7[i], 1200 + 40 * i, 45 + 3 * i,
                                      0.02 + 0.002 * i) for i in range(0, 7, 2)])
write(C, 'all.defocus', ['37 0 0. 0. 0. 3'] + [
    '%d %d %.2f %.2f %d %d %.4f %.1f %.4f' % (i + 1, i + 1, a7[i], a7[i], 1500 + 40 * i,
                                              1400 + 30 * i, -80 + 15 * i, 60 + 2 * i, 0.03)
    for i in range(0, 7, 3)])
write(C, 'x7.xf', ['%12.7f%12.7f%12.7f%12.7f%12.3f%12.3f' % (
    np.cos(t), -np.sin(t), np.sin(t), np.cos(t), 1.5 * i - 4, 0.5 * i)
    for i, t in enumerate(np.radians(-2 + 0.5 * np.arange(7)))])
write(C, 'bound.info', ['0 0 48 4 2', 'b1.mrc', '-1 0 2 -1', 'b2.mrc', '3 0 -1 0'])

M = 'mtffilter'
stack(M, 'f48', 48, 40, 7, 'float')
# mtffilter runs on ctfphaseflip's f48.mrc instead (one file, not two); the
# stack is still drawn so the random stream for the stacks below is unchanged.
os.remove(os.path.join(HERE, M, 'f48.mrc'))
stack(M, 'f33', 33, 29, 5, 'float')
stack(M, 's40', 40, 36, 4, 'short')
stack(M, 'f34', 34, 30, 6, 'float')
subprocess.run([REF + '/clip/clip', 'fft', '-3d', os.path.join(HERE, M, 'f34.mrc'),
                os.path.join(HERE, M, 'fft.mrc')], env=env, check=True, stdout=subprocess.DEVNULL)
write(M, 'mtf.dat', ['%.4f %.5f' % (k * 0.026, np.exp(-3 * k * 0.026)) for k in range(20)])
write(M, 'dose1.txt', ['%.2f' % (2.0 + 0.1 * i) for i in range(7)])
prior = np.concatenate([[0], np.cumsum(2.0 + 0.1 * np.arange(7))])
write(M, 'dose2.txt', ['%.2f %.2f' % (prior[i], 2.0 + 0.1 * i) for i in range(7)])
write(M, 'dose3.txt', ['%.2f %.2f' % (prior[i], prior[i + 1]) for i in range(7)])
lines = ['PixelSpacing = 1', 'ImageFile = f48.mrc', 'ImageSize = 48 40', 'DataMode = 2', '',
         '[T = SerialEM: test]', '']
for i in range(7):
    lines += ['[ZValue = %d]' % i, 'TiltAngle = %.2f' % (-45 + 15 * i),
              'ExposureDose = %.3f' % (2.0 + 0.1 * i), 'PriorRecordDose = %.3f' % prior[i], '']
write(M, 'f48.mrc.mdoc', lines)
