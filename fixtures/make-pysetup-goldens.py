#!/usr/bin/env python3
"""Regenerate the native goldens for one of the single-axis setup scripts
(copytomocoms, makecomfile, splittilt, alignlog, chunksetup, tomocleanup) or
of the dual-axis combine scripts (setupcombine, splitcombine, collectmmm,
b3dremove, dualvolmatch, matchorwarp, autopatchfit), and for matchrotpairs and
sirtsetup.

Usage: make-pysetup-goldens.py <script>        (called by make-<script>-goldens.sh)

Each fixtures/<script>/cases.tsv row is
    name<TAB>inputs<TAB>env<TAB>arguments
where inputs is a space-separated list of `src:dst` copies from
fixtures/<script>/inputs (`-` for none; a `src` ending in `/` copies a whole
directory's files), env is a space-separated list of `NAME=value` (`-` for
none) and arguments are split on blanks.  The native Python script
(IMOD/pysrc/<script>) runs in a fresh directory holding those inputs, with
the native reference programs on PATH (from REF, default
/tmp/imod-reference-build) and IMOD_DIR=$REF.  Recorded per case:
golden/<name>.rc, .out (stdout through a pipe) and .err, golden/<name>.files
(every file left in the directory), and golden/<name>/
holding every file present afterwards that is not an unchanged input.  For
chunksetup the output of the native `tomopieces` run is also recorded, as
golden/<name>.tomopieces, because the test replaces that external program;
the same is done for setupcombine.

Inputs for copytomocoms, and the makecomfile/tomocleanup datasets, are made
first from deterministic numpy content with the native raw2mrc and the native
copytomocoms (see make_inputs below).
"""
import os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.environ.get('REF', '/tmp/imod-reference-build')
SCRIPT = sys.argv[1]
HERE = os.path.join(ROOT, 'fixtures', SCRIPT)
PROGS = ['flib/image/header', 'flib/image/alterheader', 'clip/clip', 'imodutil/imodinfo',
         'flib/image/montagesize', 'flib/image/goodframe', 'flib/image/extracttilts',
         'flib/image/extractpieces', 'flib/image/extractmagrad', 'flib/image/tomopieces',
         'mrc/raw2mrc', 'flib/tiltalign/tiltalign', 'flib/distort/xf2rotmagstr']


def native_env(bindir):
    env = dict(os.environ)
    env.update({'AUTODOC_DIR': REF + '/autodoc', 'LD_LIBRARY_PATH': REF + '/buildlib',
                'IMOD_DIR': REF, 'PYTHONPATH': ROOT + '/IMOD/pysrc',
                'PATH': bindir + ':' + os.environ['PATH'], 'OMP_NUM_THREADS': '1'})
    for k in ('IMOD_OUTPUT_FORMAT', 'TEST_NAMING_STYLE', 'TEST_USE_PCM_FOR_COM',
              'PARALLEL_BOUNDARY_SIZE', 'RUNCMD_VERBOSE'):
        env.pop(k, None)
    return env


def header_only(path):
    """Cut an MRC file back to its header and extended header.  These scripts
    only ever read an image's header (size, mode, pixel spacing, tilt angles
    from the extended header) -- through `header`, `getmrcsize` and friends --
    never its pixels, so the data section is dead weight; regenerating the
    goldens from header-only inputs gave byte-identical native outputs
    (fixtures/README.md, "Inputs").  The same idea as IMOD's own header-only
    stubs in IMOD/Etomo/unitTestData."""
    import struct
    with open(path, 'r+b') as f:
        head = f.read(1024)
        f.truncate(1024 + struct.unpack('<i', head[92:96])[0])


def prune_unused_inputs():
    """Delete generated inputs that no row of cases.tsv (pruned `#full` rows
    included) copies in."""
    used = set()
    for row in open(os.path.join(HERE, 'cases.tsv')):
        if row.startswith('#full\t'):
            row = row[len('#full\t'):]
        if row.startswith('#') or not row.strip():
            continue
        inputs = row.rstrip('\n').split('\t')[1]
        if inputs != '-':
            used.update(item.split(':')[0].rstrip('/') for item in inputs.split())
    inp = os.path.join(HERE, 'inputs')
    for fn in os.listdir(inp):
        if fn not in used and os.path.isfile(os.path.join(inp, fn)):
            os.remove(os.path.join(inp, fn))


def make_inputs(env):
    """Deterministic inputs shared by the scripts' fixtures."""
    import numpy as np
    inp = os.path.join(ROOT, 'fixtures', 'copytomocoms', 'inputs')
    os.makedirs(inp, exist_ok=True)
    r = np.random.default_rng(5)
    work = tempfile.mkdtemp()
    for old in ('b.st', 's.st'):  # else raw2mrc leaves a `~` backup behind
        if os.path.exists(os.path.join(inp, old)):
            os.remove(os.path.join(inp, old))
    r.integers(0, 255, (6, 96, 128)).astype(np.uint8).tofile(work + '/b.raw')
    r.normal(-2000, 300, (6, 96, 128)).astype(np.int16).tofile(work + '/s.raw')
    subprocess.run(['raw2mrc', '-x', '128', '-y', '96', '-z', '6', '-t', 'byte', work + '/b.raw',
                    inp + '/b.st'], env=env, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['raw2mrc', '-x', '128', '-y', '96', '-z', '6', '-t', 'short', work + '/s.raw',
                    inp + '/s.st'], env=env, check=True, stdout=subprocess.DEVNULL)
    header_only(inp + '/b.st')
    header_only(inp + '/s.st')
    shutil.rmtree(work)
    open(inp + '/b.rawtlt', 'w').write(''.join('%g\n' % (-50 + 20 * i) for i in range(6)))
    shutil.copy(inp + '/b.rawtlt', inp + '/s.rawtlt')
    open(inp + '/change.adoc', 'w').write(
        'comparam.tilt.tilt.THICKNESS = 300\n'
        'comparam.track.beadtrack.LocalAreaTracking = 1\n'
        'comparam.newst.newstack.ExpandByFactor = 1.5\n'
        'comparam.newst.newstack.SizeToOutputInXandY = 400,400\n'
        'comparam.xcorr.tiltxcorr.FilterRadius2 = 0.2\n'
        'setupset.copyarg.voltage = 300\n'
        'setupset.copyarg.Cs = 2.7\n'
        'setupset.copyarg.defocus = 4000\n')
    # The makecomfile and tomocleanup datasets: a native copytomocoms setup of b.st
    for target in ('makecomfile', 'tomocleanup'):
        ds = os.path.join(ROOT, 'fixtures', target, 'inputs', 'dataset')
        shutil.rmtree(ds, ignore_errors=True)
        os.makedirs(ds)
        shutil.copy(inp + '/b.st', ds + '/b.st')
        shutil.copy(inp + '/b.rawtlt', ds + '/b.rawtlt')
        subprocess.run(['python3', ROOT + '/IMOD/pysrc/copytomocoms', '-name', 'b', '-pixel',
                        '1', '-gold', '5', '-rotation', '-85.3', '-userawtlt'], cwd=ds, env=env,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def make_alignlog_inputs(env):
    """Alignment logs from the native tiltalign on fixtures/tiltalign's models."""
    inp = os.path.join(HERE, 'inputs')
    os.makedirs(inp, exist_ok=True)
    cases = {}
    for row in open(os.path.join(ROOT, 'fixtures', 'tiltalign', 'cases.tsv')):
        if not row.startswith('#') and '\t' in row:
            name, args = row.rstrip('\n').split('\t')
            cases[name] = args
    runs = {'g1_basic': cases['g1_basic'],
            'g1_robust_cv': cases['g1_basic'].replace('-CrossValidate 0', '-CrossValidate 1')
            + ' -RobustFitting 1',
            'g3_xtilt_beam': cases['g3_xtilt_beam'],
            # 2 x 2 local areas rather than g2_local's 4 x 4: the same log
            # sections, a third of the size.
            'g2_local_robust_cv': cases['g2_local'].replace('-CrossValidate 0', '-CrossValidate 1')
            .replace('-TargetPatchSizeXandY 700,700', '-TargetPatchSizeXandY 1200,1200')
            + ' -RobustFitting 1'}
    for name, args in runs.items():
        work = tempfile.mkdtemp()
        for fn in os.listdir(os.path.join(ROOT, 'fixtures', 'tiltalign')):
            if fn[:3] in ('g1.', 'g2.', 'g3.'):
                shutil.copy(os.path.join(ROOT, 'fixtures', 'tiltalign', fn), work)
        p = subprocess.run(['tiltalign'] + args.split(), cwd=work, env=env, capture_output=True)
        open(os.path.join(inp, name + '.log'), 'wb').write(p.stdout + p.stderr)
        shutil.rmtree(work)
    # The vendored IMOD/Etomo/unitTestData/aligna.log (an older format) is
    # read in place: cases.tsv names it as ../../../IMOD/Etomo/unitTestData/.


def make_combine_inputs(env):
    """Small dual-axis inputs for the combine scripts: byte tomograms (X by
    thickness by Y, 5 A pixels), raw stacks (2.5 A pixels, so binning 2),
    transforms, tilt angles, aligned stacks and tilt command files; for
    splitcombine, volcombine command files made by the native setupcombine;
    for collectmmm, chunk logs and an image."""
    import math
    import numpy as np
    inp = os.path.join(HERE, 'inputs')
    shutil.rmtree(inp, ignore_errors=True)
    os.makedirs(inp)
    r = np.random.default_rng(11)
    work = tempfile.mkdtemp()

    def mrc(name, shape, pixel, dtype='byte'):
        raw = work + '/x.raw'
        if dtype == 'byte':
            r.integers(0, 255, shape[::-1]).astype(np.uint8).tofile(raw)
        else:
            r.normal(10, 3, shape[::-1]).astype(np.float32).tofile(raw)
        subprocess.run(['raw2mrc', '-x', str(shape[0]), '-y', str(shape[1]), '-z', str(shape[2]),
                        '-t', dtype, raw, name], env=env, check=True, stdout=subprocess.DEVNULL)
        if pixel:
            subprocess.run([REF + '/flib/image/alterheader', '-PixelSize', '%g,%g,%g' % ((pixel,) * 3),
                            name], env=env, check=True, stdout=subprocess.DEVNULL)
        if SCRIPT != 'collectmmm':
            header_only(name)

    if SCRIPT == 'collectmmm':
        mrc(inp + '/img.mrc', (32, 30, 3), 0)
        for num, vals in ((1, '-1.5 20 3.25 1000'), (2, '-3 17.5 4 3000'), (3, '0.5 25 1e1 500')):
            open(inp + '/r-%03d.log' % num, 'w').write('some text\npixels= %s\nmore\n' % vals)
        open(inp + '/three.log', 'w').write('pixels= 1 2 3\n')
        open(inp + '/bad.log', 'w').write('pixels= 1 2 x 4\n')
        shutil.rmtree(work)
        return
    if SCRIPT == 'b3dremove':
        for name in ('a1.log', 'a2.log', 'b.txt'):
            open(os.path.join(inp, name), 'w').write('x\n')
        os.makedirs(inp + '/d1/x')
        open(inp + '/d1/x/y', 'w').write('1\n')
        shutil.rmtree(work)
        return
    for ax in 'ab':
        mrc(inp + '/g%s.rec' % ax, (64, 20, 64), 5)
        mrc(inp + '/g%s.st' % ax, (32, 30, 11), 2.5)
        mrc(inp + '/g%s.ali' % ax, (16, 60, 11), 0)
        tilts = [-50 + 10 * i for i in range(11)]
        tilts[5] = 0.3 if ax == 'a' else -0.7
        open(inp + '/g%s.tlt' % ax, 'w').write(''.join('%8.2f\n' % t for t in tilts))
        lines = []
        for i in range(11):
            ang = math.radians((-12.5 if ax == 'a' else 77.4) + 0.1 * i)
            c, s = math.cos(ang), math.sin(ang)
            lines.append('%12.7f%12.7f%12.7f%12.7f%12.3f%12.3f\n' % (c, -s, s, c, 1.5 * i, -0.5 * i))
        open(inp + '/g%s.xf' % ax, 'w').write(''.join(lines))
    tilt = open(REF + '/com/tilt.com').read()
    for ax, xaxis, extra, binned in (('a', '1.5', 'OFFSET -0.3\nSHIFT 0 4\n', '2'),
                                     ('b', '-2.25', 'OFFSET 0.7\nSHIFT 0. 3.\n', '1')):
        t = tilt.replace('g5a', 'g' + ax).replace('XAXISTILT 0.', 'XAXISTILT ' + xaxis)
        open(inp + '/tilt%s.com' % ax, 'w').write(t.replace('IMAGEBINNED 1', 'IMAGEBINNED ' + binned)
                                                  + extra)
        open(inp + '/ctilt%s.com' % ax, 'w').write(
            t.replace('IMAGEBINNED 1', 'IMAGEBINNED ' + binned) + 'SHIFT 0 1\nSLICE 25 35\n')
    open(inp + '/change.adoc', 'w').write(
        'comparam.solvematch.solvematch.MaximumResidual = 12\n'
        'comparam.matchorwarp.matchorwarp.WarpLimits=0.3,0.4\n'
        'comparam.volcombine.combinefft.ReductionFraction=0.1\n')
    open(inp + '/empty', 'w').write('')
    if SCRIPT == 'splitcombine':
        # volcombine files from the native setupcombine: one piece, chunked,
        # a /tmp temporary directory, /usr/tmp, and naming style 1
        for f in ('ga.rec', 'gb.rec', 'ga.st', 'gb.st', 'tilta.com', 'tiltb.com'):
            shutil.copy(inp + '/' + f, work + '/' + f)
        base = ['python3', ROOT + '/IMOD/pysrc/setupcombine', '-name', 'g', '-surfaces', '1',
                '-zlimits', '3,17', '-stackext', 'st']
        big = []
        for name, extra in (('vc_basic', []), ('vc_chunk', ['-chunked', '1']),
                            ('vc_tmp', ['-tempdir', '/tmp']), ('vc_lowrad', ['-lowradius', '0.1'])):
            subprocess.run(base + extra, cwd=work, env=env, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            shutil.copy(work + '/volcombine.com', inp + '/' + name + '.com')
        text = open(inp + '/vc_tmp.com').read()
        open(inp + '/vc_usrtmp.com', 'w').write(text.replace('/tmp/', '/usr/tmp/'))
        # Several pieces: a header-only 512 x 100 x 512 volume, sparse on disk
        import struct
        for ax in 'ab':
            h = bytearray(1024)
            struct.pack_into('<3i i 3i 3i 3f 3f 3i 3f', h, 0, 512, 100, 512, 0, 0, 0, 0, 512, 100,
                             512, 2560., 500., 2560., 90., 90., 90., 1, 2, 3, 0., 1., 0.5)
            h[208:212] = b'MAP '
            h[212:216] = bytes([0x44, 0x44, 0, 0])
            with open(work + '/g%s.rec' % ax, 'wb') as f:
                f.write(h)
                f.truncate(1024 + 512 * 100 * 512)
        for name, extra in (('vc_pieces', []), ('vc_pieces_chunk', ['-chunked', '1'])):
            subprocess.run(base + extra, cwd=work, env=env, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            shutil.copy(work + '/volcombine.com', inp + '/' + name + '.com')
        for f in os.listdir(inp):
            if not f.endswith('.com') or not f.startswith('vc_'):
                os.remove(os.path.join(inp, f))
    if SCRIPT in ('dualvolmatch', 'matchorwarp', 'autopatchfit'):
        for f in ('ga.st', 'gb.st', 'ga.ali', 'gb.ali', 'ga.tlt', 'gb.tlt', 'ga.xf', 'gb.xf',
                  'tilta.com', 'tiltb.com', 'ctilta.com', 'ctiltb.com', 'change.adoc'):
            os.remove(os.path.join(inp, f))
        open(inp + '/patch.out', 'w').write('1 positions\n')
        open(inp + '/solve.xf', 'w').write('1 0 0 0\n0 1 0 0\n0 0 1 0\n')
        pc = open(REF + '/com/patchcorr.com').read().replace('g5a', 'ga').replace('g5b', 'gb')
        open(inp + '/patchcorr.com', 'w').write(pc)
        open(inp + '/pc_noout.com', 'w').write('$corrsearch3d -StandardInput\nFileToAlign gb.mat\n')
        open(inp + '/pc_nomat.com', 'w').write('$corrsearch3d -StandardInput\nOutputFile patch.out\n')
        open(inp + '/pc_nosizes.com', 'w').write('$corrsearch3d -StandardInput\nPatchSizeXYZ 32,14,32\n')
        mow = open(REF + '/com/matchorwarp.com').read().replace('g5a', 'ga').replace('g5b', 'gb')
        open(inp + '/matchorwarp.com', 'w').write(mow)
        open(inp + '/mow_nowarp.com', 'w').write('$matchorwarp -StandardInput\nInputVolume gb.rec\n')
    shutil.rmtree(work)


def make_sirtsetup_inputs(env):
    """Inputs for sirtsetup: header-only aligned stacks (sirtsetup reads only
    their sizes and mode), tilt command files for the internal-SIRT, vertical
    slice (X-axis tilt), local-alignment (external SIRT with LOG and SCALE)
    and GPU cases, X-tilt files, and header-only "srec"/"vsr"/"sint" files of
    the reconstruction's X/Z size for the resume and cleanup cases."""
    import numpy as np
    inp = os.path.join(HERE, 'inputs')
    shutil.rmtree(inp, ignore_errors=True)
    os.makedirs(inp)
    work = tempfile.mkdtemp()
    np.zeros((5, 48, 64), np.uint8).tofile(work + '/a.raw')
    subprocess.run(['raw2mrc', '-x', '64', '-y', '48', '-z', '5', '-t', 'byte', work + '/a.raw',
                    inp + '/ts.ali'], env=env, check=True, stdout=subprocess.DEVNULL)
    header_only(inp + '/ts.ali')
    np.zeros((48, 1, 64), np.uint8).tofile(work + '/r.raw')
    subprocess.run(['raw2mrc', '-x', '64', '-y', '1', '-z', '48', '-t', 'byte', work + '/r.raw',
                    inp + '/rec.mrc'], env=env, check=True, stdout=subprocess.DEVNULL)
    header_only(inp + '/rec.mrc')
    shutil.rmtree(work)
    tilt = ('# Command file to run Tilt\n#\n$tilt -StandardInput\nInputProjections ts.ali\n'
            'OutputFile ts.rec\nIMAGEBINNED 1\nTILTFILE ts.tlt\nXTILTFILE ts.xtilt\n'
            'THICKNESS 30\nRADIAL .35 .035\nFalloffIsTrueSigma 1\nXAXISTILT 0.\nLOG 0\n'
            'SCALE 0 1000\nPERPENDICULAR\nMODE 2\nFULLIMAGE 64 48\nSUBSETSTART 0 0\n'
            'AdjustOrigin 1\n$if (-e ./savework) ./savework\n')
    open(inp + '/tilt.com', 'w').write(tilt)
    open(inp + '/xtilt.com', 'w').write(tilt.replace('XAXISTILT 0.', 'XAXISTILT 3.5'))
    open(inp + '/local.com', 'w').write(tilt.replace('XAXISTILT 0.', 'LOCALFILE tslocal.xf'))
    open(inp + '/noscale.com', 'w').write(
        tilt.replace('XAXISTILT 0.', 'LOCALFILE tslocal.xf').replace('SCALE 0 1000\n', ''))
    open(inp + '/gpu.com', 'w').write(tilt.replace('MODE 2', 'MODE 2\nUseGPU 0'))
    open(inp + '/ts.xtilt', 'w').write('0\n' * 5)
    open(inp + '/xtvar.xtilt', 'w').write('1.5\n' * 4 + '2.0\n')
    open(inp + '/xtsame.xtilt', 'w').write('1.5\n' * 5)
    open(inp + '/empty', 'w').write('')
    open(inp + '/ts.tlt', 'w').write(''.join('%.2f\n' % (-40 + 20 * i) for i in range(5)))


def make_matchrotpairs_inputs(env):
    """Two small byte tilt series for matchrotpairs: A is 5 views of a smooth
    random field drifting slowly from view to view; B holds the same views
    one step later, rotated by 90 degrees (np.rot90, no interpolation) and
    shifted by a few pixels.  Plus a zero-stretch distortion field for 64 x 64
    images and a stale output file for the backup case."""
    import numpy as np
    inp = os.path.join(HERE, 'inputs')
    shutil.rmtree(inp, ignore_errors=True)
    os.makedirs(inp)
    r = np.random.default_rng(23)
    n = 64
    field = r.normal(0, 1, (n * 2, n * 2))
    ky = np.fft.fftfreq(n * 2)[:, None]
    kx = np.fft.fftfreq(n * 2)[None, :]
    field = np.real(np.fft.ifft2(np.fft.fft2(field) * np.exp(-(kx ** 2 + ky ** 2) / (2 * 0.04 ** 2))))
    views = []
    for v in range(6):
        view = field[16 + 2 * v:16 + 2 * v + n, 20 + v:20 + v + n]
        views.append(view)
    lo, hi = min(v.min() for v in views), max(v.max() for v in views)
    scale = lambda v: np.clip((v - lo) / (hi - lo) * 250 + 2, 0, 255).astype(np.uint8)
    a = np.stack([scale(v) for v in views[0:5]])
    b = np.stack([np.roll(np.rot90(scale(v)), (3, -2), axis=(0, 1)) for v in views[1:6]])
    work = tempfile.mkdtemp()
    for name, arr in (('a.st', a), ('b.st', b)):
        arr.tofile(work + '/x.raw')
        subprocess.run(['raw2mrc', '-x', str(n), '-y', str(n), '-z', '5', '-t', 'byte',
                        work + '/x.raw', inp + '/' + name], env=env, check=True,
                       stdout=subprocess.DEVNULL)
    shutil.rmtree(work)
    vals = r.uniform(-1., 1., 2 * 8 * 8)
    with open(inp + '/d64.idf', 'w') as f:
        f.write('2\n64 64 1 1 1.000000\n4.000000 8.000000 8 4.000000 8.000000 8\n')
        for i in range(0, len(vals), 8):
            f.write('  '.join('%.3f' % x for x in vals[i:i + 8]) + '\n')
    open(inp + '/old.xf', 'w').write('1 0 0 1 0 0\n')


def populate(work, inputs):
    placed = {}
    if inputs == '-':
        return placed
    for item in inputs.split():
        src, dst = item.split(':')
        s = os.path.join(HERE, 'inputs', src)
        if src.endswith('/'):
            for fn in sorted(os.listdir(s)):
                if os.path.isdir(os.path.join(s, fn)):
                    continue
                t = os.path.join(work, dst, fn)
                os.makedirs(os.path.dirname(t), exist_ok=True)
                shutil.copy(os.path.join(s, fn), t)
                placed[os.path.relpath(t, work)] = open(t, 'rb').read()
            for fn in sorted(os.listdir(s)):
                if os.path.isdir(os.path.join(s, fn)):
                    os.makedirs(os.path.join(work, dst, fn), exist_ok=True)
                    for f2 in sorted(os.listdir(os.path.join(s, fn))):
                        t = os.path.join(work, dst, fn, f2)
                        shutil.copy(os.path.join(s, fn, f2), t)
                        placed[os.path.relpath(t, work)] = open(t, 'rb').read()
        else:
            t = os.path.join(work, dst)
            os.makedirs(os.path.dirname(t), exist_ok=True)
            shutil.copy(s, t)
            placed[dst] = open(t, 'rb').read()
    return placed


def main():
    bindir = tempfile.mkdtemp()
    for p in PROGS:
        os.symlink(os.path.join(REF, p), os.path.join(bindir, os.path.basename(p)))
    if SCRIPT == 'matchrotpairs':
        for p in ('flib/image/newstack', 'flib/image/xfsimplex', 'flib/image/xfproduct',
                  'imodutil/tiltxcorr'):
            os.symlink(os.path.join(REF, p), os.path.join(bindir, os.path.basename(p)))
    env = native_env(bindir)
    if SCRIPT == 'matchrotpairs':
        make_matchrotpairs_inputs(env)
    if SCRIPT == 'copytomocoms':
        make_inputs(env)
    if SCRIPT == 'sirtsetup':
        # sirtsetup runs `splittilt`, a Python script, through the shell
        wrap = os.path.join(bindir, 'splittilt')
        open(wrap, 'w').write('#!/bin/sh\nexec python3 %s/IMOD/pysrc/splittilt "$@"\n' % ROOT)
        os.chmod(wrap, 0o755)
        make_sirtsetup_inputs(env)
    if SCRIPT == 'alignlog':
        make_alignlog_inputs(env)
    if SCRIPT in ('setupcombine', 'splitcombine', 'collectmmm', 'b3dremove', 'dualvolmatch',
                  'matchorwarp', 'autopatchfit'):
        make_combine_inputs(env)
        prune_unused_inputs()
    golden = os.path.join(HERE, 'golden')
    shutil.rmtree(golden, ignore_errors=True)
    os.makedirs(golden)
    for row in open(os.path.join(HERE, 'cases.tsv')):
        # FULL=1 also runs the rows pruned from the ordinary suite (`#full<TAB>row`).
        if os.environ.get('FULL') and row.startswith('#full\t'):
            row = row[len('#full\t'):]
        if row.startswith('#') or not row.strip():
            continue
        name, inputs, envs, args = row.rstrip('\n').split('\t')
        work = tempfile.mkdtemp()
        placed = populate(work, inputs)
        cenv = dict(env)
        if envs != '-':
            for kv in envs.split():
                k, v = kv.split('=', 1)
                cenv[k] = v
        argv = [] if args == '-' else args.split()
        if SCRIPT in ('chunksetup', 'setupcombine'):
            # Record what the native tomopieces prints for this case
            wrap = os.path.join(bindir, 'tomopieces')
            os.remove(wrap)
            open(wrap, 'w').write('#!/bin/sh\n%s/flib/image/tomopieces "$@" | tee %s/.tomopieces\n'
                                  % (REF, work))
            os.chmod(wrap, 0o755)
        p = subprocess.run(['python3', os.path.join(ROOT, 'IMOD/pysrc', SCRIPT)] + argv, cwd=work,
                           env=cenv, capture_output=True)
        open(os.path.join(golden, name + '.rc'), 'w').write('%d\n' % p.returncode)
        open(os.path.join(golden, name + '.out'), 'wb').write(p.stdout)
        open(os.path.join(golden, name + '.err'), 'wb').write(p.stderr)
        if os.path.exists(work + '/.tomopieces'):
            shutil.move(work + '/.tomopieces', os.path.join(golden, name + '.tomopieces'))
        remaining = []
        for rootdir, dirs, files in os.walk(work):
            for fn in files:
                remaining.append(os.path.relpath(os.path.join(rootdir, fn), work))
        open(os.path.join(golden, name + '.files'), 'w').write(
            ''.join(f + '\n' for f in sorted(remaining)))
        for rootdir, dirs, files in os.walk(work):
            for fn in files:
                f = os.path.join(rootdir, fn)
                rel = os.path.relpath(f, work)
                data = open(f, 'rb').read()
                if placed.get(rel) == data:
                    continue
                t = os.path.join(golden, name, rel)
                os.makedirs(os.path.dirname(t), exist_ok=True)
                open(t, 'wb').write(data)
                shutil.copymode(f, t)
            for d in dirs:
                os.makedirs(os.path.join(golden, name, os.path.relpath(os.path.join(rootdir, d), work)),
                            exist_ok=True)
        shutil.rmtree(work)
        if SCRIPT in ('chunksetup', 'setupcombine'):
            os.remove(os.path.join(bindir, 'tomopieces'))
            os.symlink(os.path.join(REF, 'flib/image/tomopieces'), os.path.join(bindir, 'tomopieces'))
        print(name, p.returncode)
    shutil.rmtree(bindir)


main()
